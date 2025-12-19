import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import shutil
import time
import random
import re
import torch
import chromadb
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= 配置区域 =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DB_PATH = "./reflexion_full_db"
CHUNK_SIZE = 64  # 处理单元大小
MAX_NEW_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.90 
TARGET_ACCURACY = 92.0  # [修改] 目标准确率，达到后停止训练
MAX_EPOCHS = 5        # [修改] 最大训练轮数，防止死循环

# ================= 1. 记忆管理器 (保持不变) =================
class MemoryManager:
    def __init__(self, reset=False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        # 内存缓存 Stats
        self.skill_stats = {} 
        self.current_step = 0 # 全局计数器

    def batch_retrieve(self, query_embeddings, top_k=3, threshold=0.5):
        count = self.collection.count()
        if count == 0: return [[] for _ in range(len(query_embeddings))]
        
        real_k = min(top_k, count)
        results_list = [] 
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=real_k)
            for i in range(len(query_embeddings)):
                sample_rules = []
                if results['ids'][i]:
                    for j in range(len(results['ids'][i])):
                        dist = results['distances'][i][j]
                        if dist < threshold:
                            doc = results['documents'][i][j]
                            sid = results['ids'][i][j]
                            sample_rules.append((doc, dist, sid))
                results_list.append(sample_rules)
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return [[] for _ in range(len(query_embeddings))]
        return results_list

    def add_experience_batch(self, patterns_A, strategies_B, embeddings_A):
        if not patterns_A: return
        unique_patterns = []
        unique_strategies = []
        unique_embeddings = []
        
        for i, emb in enumerate(embeddings_A):
            try:
                existing = self.collection.query(query_embeddings=[emb], n_results=1)
                if existing['ids'] and existing['ids'][0] and existing['distances'][0][0] < 0.15: 
                    exist_id = existing['ids'][0][0]
                    if exist_id in self.skill_stats:
                        self.skill_stats[exist_id]['score'] = min(1.0, self.skill_stats[exist_id]['score'] + 0.05)
                    continue 
            except:
                pass
            unique_patterns.append(patterns_A[i])
            unique_strategies.append(strategies_B[i])
            unique_embeddings.append(embeddings_A[i])

        if not unique_patterns: return

        new_ids = [f"rule_{int(time.time())}_{i}_{random.randint(0,999)}" for i in range(len(unique_patterns))]
        metadatas = [{"pattern": p} for p in unique_patterns]
        
        self.collection.add(
            ids=new_ids,
            embeddings=unique_embeddings,
            documents=unique_strategies,
            metadatas=metadatas
        )
        
        for sid in new_ids:
            self.skill_stats[sid] = {
                "score": 0.5, "usage": 0, "history_correct": 0, 
                "created_step": self.current_step, "is_probation": True 
            }

    def update_scores_batch(self, usage_data_batch, is_correct_list, model_outputs):
        self.current_step += 1 
        for i, used_rules in enumerate(usage_data_batch):
            if not used_rules: continue
            is_correct = is_correct_list[i]
            output_text = model_outputs[i]
            
            for rule_item in used_rules:
                sid, _, content = rule_item
                if sid not in self.skill_stats:
                    self.skill_stats[sid] = {"score": 0.5, "usage": 0, "history_correct": 0, "created_step": 0, "is_probation": False}
                stats = self.skill_stats[sid]
                
                if len(content) > 10:
                    fingerprint = content[:10] 
                    if fingerprint not in output_text and not is_correct:
                        continue 

                stats['usage'] += 1
                if is_correct:
                    stats['history_correct'] += 1
                    stats['score'] = min(1.0, stats['score'] + 0.1)
                    if stats.get('is_probation') and stats['score'] > 0.6:
                        stats['is_probation'] = False
                else:
                    penalty = 0.2
                    if stats['history_correct'] > 10: penalty = 0.1
                    if stats['history_correct'] > 50: penalty = 0.05
                    stats['score'] = max(0.0, stats['score'] - penalty)

    def prune_db(self, min_usage=5, threshold=0.3):
        ids_to_delete = []
        decay_rate = 0.01 
        for sid, stats in list(self.skill_stats.items()):
            if stats['score'] < 0.95:
                stats['score'] -= decay_rate
            if stats.get('is_probation', False):
                if stats['score'] < 0.4:
                    ids_to_delete.append(sid)
                    continue
            if stats['usage'] >= min_usage and stats['score'] < threshold:
                ids_to_delete.append(sid)
                continue
            age = self.current_step - stats.get('created_step', 0)
            if age > 1000 and stats['usage'] < 2:
                ids_to_delete.append(sid)

        if ids_to_delete:
            print(f"✂️ [淘汰] 清理 {len(ids_to_delete)} 条低分规则 (剩余: {len(self.skill_stats) - len(ids_to_delete)})")
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.skill_stats[sid]
            return len(ids_to_delete)
        return 0

# ================= 2. vLLM 进化训练器 (主要修改区域) =================
class ReflexionTrainerFull:
    def __init__(self):
        print("🚀 初始化训练环境 (vLLM Accelerated Mode)...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            max_model_len=2048,
            download_dir=HF_CACHE_DIR
        )
        
        self.params_inference = SamplingParams(
            temperature=0.5, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        self.params_reflection = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        self.params_verify = SamplingParams(
            temperature=0.1, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        print("   -> Loading Embedder (Force CPU to save GPU memory)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu", cache_folder=HF_CACHE_DIR)
        
        self.memory = MemoryManager(reset=True)

        self.rag_log_path = "rag_usage_log.jsonl"
        open(self.rag_log_path, "w").close() 

    def batch_generate_vllm(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def has_specific_numbers(self, text, question_text):
        q_nums = set(re.findall(r'\b\d+\b', question_text))
        r_nums = set(re.findall(r'\b\d+\b', text))
        whitelist = {'1', '2', '3', '4', '10', '100', '180', '360'} 
        for n in r_nums:
            if n in q_nums and n not in whitelist:
                return True 
        return False

    def clean_rule_text(self, text):
        text = text.replace("```markdown", "").replace("```", "").strip()
        patterns_to_remove = [
            r"^Sure,.*?\n", r"^Here is.*?\n", r"^Certainly.*?\n",
            r"^To summarize.*?\n", r"You are an AI assistant.*?",
            r"Assistant:.*?", r"Great job!.*?", r"#### \d+"
        ]
        for pat in patterns_to_remove:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if text.count("**Trigger (A)**") > 1:
            first_occurrence = text.find("**Trigger (A)**", text.find("**Trigger (A)**") + 1)
            text = text[:first_occurrence]
        if text.count("**Strategy (B)**") > 1:
            first_occurrence = text.find("**Strategy (B)**", text.find("**Strategy (B)**") + 1)
            text = text[:first_occurrence]
        return text.strip()

    def parse_reflection(self, text):
        cleaned_text = self.clean_rule_text(text)
        trigger_match = re.search(r"(?:\*\*Trigger \(A\)\*\*|Trigger \(A\)|Trigger):?\s*(.*?)(?=\n|(?:\*\*Strategy)|$)", cleaned_text, re.DOTALL | re.IGNORECASE)
        strategy_match = re.search(r"(?:\*\*Strategy \(B\)\*\*|Strategy \(B\)|Strategy):?\s*(.*?)(?=\n\s*(?:\*\*Trigger)|$)", cleaned_text, re.DOTALL | re.IGNORECASE)
        
        if trigger_match and strategy_match:
            t_text = trigger_match.group(1).strip()
            s_text = strategy_match.group(1).strip()
            t_text = re.sub(r"\s+", " ", t_text)
            s_text = re.sub(r"\s+", " ", s_text)

            if len(t_text) < 5 or len(s_text) < 5: return None
            if len(t_text) > 200: t_text = t_text[:200] 
            if "Strategy" in t_text: t_text = t_text.split("Strategy")[0].strip()
            return t_text, s_text
        return None

    def construct_prompt(self, q, context=None):
        if context:
            content = f"""
You may use the following abstract strategy ONLY if it is relevant.
Do NOT introduce new examples, sub-questions, or conditions.

[STRATEGY]
{context}

[QUESTION]
{q}

Answer step-by-step and give ONLY the final answer.
"""
        else:
            content = f"Question: {q}\nAnswer step-by-step:"
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def run_full_evolution(self):
        # 1. 准备数据
        dataset = load_dataset("gsm8k", "main")['train'].select(range(200)) 
        
        total_len = len(dataset)
        print(f"⚡️ 正在预计算 {total_len} 条问题的 Embedding (CPU Mode)...")
        
        all_questions_raw = dataset['question']
        all_answers_raw = dataset['answer']
        
        # 预计算 Embeddings (Epoch 间复用)
        all_q_embeddings = self.embedder.encode(all_questions_raw, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        print(f"🔥 开始全量进化训练 (vLLM Speedup) - 目标准确率: {TARGET_ACCURACY}%")

        epoch = 0
        best_acc = 0.0

        while best_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS:
            epoch += 1
            print(f"\n======== Epoch {epoch}/{MAX_EPOCHS} ========")
            
            # 1. Shuffle 数据索引
            indices = np.random.permutation(total_len)
            
            epoch_correct_count = 0
            epoch_total_count = 0
            
            pbar = tqdm(range(0, total_len, CHUNK_SIZE), desc=f"Epoch {epoch} Training")
            
            for chunk_start in pbar:
                chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
                current_batch_indices = indices[chunk_start:chunk_end]
                
                # 获取当前 Batch 的原始数据
                chunk_questions = [all_questions_raw[i] for i in current_batch_indices]
                chunk_answers = [all_answers_raw[i] for i in current_batch_indices]
                chunk_q_embeddings = all_q_embeddings[current_batch_indices]
                
                # ================= 阶段 1: Zero-shot 推理 =================
                zero_shot_prompts = [self.construct_prompt(q) for q in chunk_questions]
                zs_outputs = self.batch_generate_vllm(zero_shot_prompts, self.params_inference)
                
                # 评估 Zero-shot 结果
                zs_is_correct = []
                incorrect_local_indices = [] # 记录在这个 chunk 中做错的下标
                
                for idx, pred in enumerate(zs_outputs):
                    gt = chunk_answers[idx]
                    is_right = self.check_answer(pred, gt)
                    zs_is_correct.append(is_right)

                    if not is_right:
                        incorrect_local_indices.append(idx)
                    epoch_total_count += 1
                    
                chunk_final_correct = zs_is_correct[:]
                
                # ================= 阶段 2: RAG 重算 (仅针对错题) =================
                rag_usage_for_update = []
                rag_is_correct_for_update = []
                rag_outputs_for_update = []
                
                still_incorrect_indices = []   # RAG 后依然做错的下标，用于反思
                
                if incorrect_local_indices:
                    # 1. 准备错题数据
                    wrong_questions = [chunk_questions[i] for i in incorrect_local_indices]
                    wrong_embeddings = chunk_q_embeddings[incorrect_local_indices]
                    wrong_answers = [chunk_answers[i] for i in incorrect_local_indices]
                    
                    # 2. 检索规则
                    retrieved_batch = self.memory.batch_retrieve(wrong_embeddings, top_k=3, threshold=0.4)
                    
                    rag_prompts = []
                    valid_rag_indices_map = [] # 记录有规则的错题对应的是哪个 local index
                    
                    for k, q in enumerate(wrong_questions):
                        rules_list = retrieved_batch[k]
                        if rules_list:
                            context_text = "\n".join([f"- {r[0]}" for r in rules_list])
                            rag_prompts.append(self.construct_prompt(q, context_text))
                            # 记录数据以便后续 update_score
                            rag_usage_for_update.append(rules_list)
                            valid_rag_indices_map.append(k)
                        else:
                            # 如果没检索到规则，就没必要 RAG 重算了，直接视为依然错误
                            rag_usage_for_update.append([]) # 空规则占位，不参与更新但保持索引对齐
                            rag_prompts.append(None) # 占位
                            still_incorrect_indices.append(incorrect_local_indices[k])

                    # 3. 执行 RAG 推理 (只推理有 Prompts 的部分)
                    real_rag_prompts = [p for p in rag_prompts if p is not None]
                    if real_rag_prompts:
                        real_rag_outputs = self.batch_generate_vllm(real_rag_prompts, self.params_inference)
                    
                    # 4. 评估 RAG 结果并准备 Update 数据
                    output_cursor = 0
                    for k, q in enumerate(wrong_questions):
                        if rag_prompts[k] is None:
                            # 没规则，跳过 Update，直接判定为 Refection 候选
                            continue
                            
                        pred = real_rag_outputs[output_cursor]
                        output_cursor += 1
                        
                        gt = wrong_answers[k]
                        is_right = self.check_answer(pred, gt)
                        
                        # 记录 Update 数据
                        rag_is_correct_for_update.append(is_right)
                        rag_outputs_for_update.append(pred)
                        
                        if not is_right:
                            chunk_final_correct[incorrect_local_indices[k]] = True
                        else:
                            still_incorrect_indices.append(incorrect_local_indices[k])
                    
                    # 5. 更新 Memory 分数 (仅针对使用了 RAG 的错题)
                    # 注意：这里需要过滤掉 rag_usage_for_update 中的空列表，虽然 update_scores_batch 内部也会跳过空列表，但为了对齐 is_correct_list 最好清洗一下
                    clean_usage = []
                    clean_correct = []
                    clean_outputs = []

                    final_usage = [rag_usage_for_update[k] for k in range(len(wrong_questions)) if rag_prompts[k] is not None]
                    
                    if final_usage:
                        self.memory.update_scores_batch(final_usage, rag_is_correct_for_update, rag_outputs_for_update)

                # ================= 阶段 3: 反思 (仅针对 RAG 后依然错误的题) =================
                if still_incorrect_indices:
                    reflect_prompts = []
                    verify_data = [] 
                    
                    for idx in still_incorrect_indices:
                        q = chunk_questions[idx]
                        gt = chunk_answers[idx]
                        
                        p_content = f"""Task: Extract a general math rule from the incorrect problem.
STRICT FORMAT INSTRUCTION:
1. Do NOT print "Sure", "Here is", or any conversational filler.
2. Do NOT use markdown code blocks.
3. Start directly with "**Trigger (A)**".
4. Use variables like X, Y, Z instead of specific numbers.
5. STOP after providing the Strategy.
[EXAMPLE]
Problem: John buys 5 apples for $2 each. Total cost?
Solution: 5 * 2 = 10.
**Trigger (A)**: Calculating total cost from quantity and rate.
**Strategy (B)**: Total Cost = Quantity * Unit Price.
[YOUR TURN]
Problem: {q}
Solution: {gt}
Summarize the rule:"""
                        reflect_prompts.append(f"<|im_start|>user\n{p_content}<|im_end|>\n<|im_start|>assistant\n")
                        verify_data.append((q, gt))
                    
                    # 生成反思
                    reflections = self.batch_generate_vllm(reflect_prompts, self.params_reflection)
                    
                    temp_candidates = []
                    for k, text in enumerate(reflections):
                        parsed = self.parse_reflection(text)
                        if parsed:
                            p_text, s_text = parsed
                            orig_q = verify_data[k][0]
                            if self.has_specific_numbers(s_text, orig_q): continue 
                            temp_candidates.append((p_text, s_text, k))
                    
                    # 验证反思
                    if temp_candidates:
                        verify_prompts = []
                        for p_text, s_text, k in temp_candidates:
                            orig_q = verify_data[k][0]
                            vp_content = f"Rule: {s_text}\nQuestion: {orig_q}\nAnswer step-by-step:"
                            verify_prompts.append(f"<|im_start|>user\n{vp_content}<|im_end|>\n<|im_start|>assistant\n")
                        
                        verify_outputs = self.batch_generate_vllm(verify_prompts, self.params_verify)
                        
                        verified_patterns = []
                        verified_strategies = []
                        
                        for m, pred in enumerate(verify_outputs):
                            k_idx = temp_candidates[m][2]
                            orig_gt = verify_data[temp_candidates[m][2]][1]
                            if self.check_answer(pred, orig_gt):
                                verified_patterns.append(temp_candidates[m][0])
                                verified_strategies.append(temp_candidates[m][1])
                                original_global_idx = still_incorrect_indices[k_idx]
                                chunk_final_correct[original_global_idx] = True
                        
                        if verified_patterns:
                            p_embeds = self.embedder.encode(verified_patterns, convert_to_numpy=True).tolist()
                            self.memory.add_experience_batch(verified_patterns, verified_strategies, p_embeds)

                # --- 定期淘汰 ---
                if (chunk_start // CHUNK_SIZE) % 5 == 0:
                    self.memory.prune_db(threshold=0.25)

                epoch_correct_count += sum(chunk_final_correct)

                # 更新进度条信息 (这里的 Accuracy 是 Zero-shot 的准确率)
                batch_acc = sum(chunk_final_correct) / len(chunk_questions) * 100
                pbar.set_postfix({"Total Acc": f"{batch_acc:.1f}%", "DB": self.memory.collection.count()})
            
            # --- Epoch 总结 ---
            current_epoch_acc = (epoch_correct_count / epoch_total_count) * 100
            print(f"\n📊 Epoch {epoch} 完成 | 综合准确率 (ZS+RAG+Reflect): {current_epoch_acc:.2f}% (Target: {TARGET_ACCURACY}%)")
            
            if current_epoch_acc > best_acc:
                best_acc = current_epoch_acc
            
            if current_epoch_acc >= TARGET_ACCURACY:
                print(f"🎉 恭喜！达到目标准确率 ({current_epoch_acc:.2f}%)，训练完成！")
                break
            elif epoch >= MAX_EPOCHS:
                print("⚠️ 达到最大 Epoch 限制，停止训练。")
                break
            else:
                print("🔄 表现未达标，继续下一轮训练...")

    def extract_number(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    def check_answer(self, pred, gt):
        if "####" in gt:
            gold = self.extract_number(gt.split("####")[1])
        else:
            gold = self.extract_number(gt)
        pred_num = self.extract_number(pred)
        if gold is None or pred_num is None: return False
        return abs(gold - pred_num) < 1e-4

    def cleanup(self):
        print("🧹 正在清理显存和 vLLM 进程...")
        if hasattr(self, 'llm'):
            del self.llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass
        print("✅ 清理完成！")

if __name__ == "__main__":
    trainer = None
    try:
        trainer = ReflexionTrainerFull()
        trainer.run_full_evolution()
    except KeyboardInterrupt:
        print("\n🛑 用户强制停止训练")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise e
    finally:
        if trainer is not None:
            trainer.cleanup()