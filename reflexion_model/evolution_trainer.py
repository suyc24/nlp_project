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
        # 为了测试流畅，建议只取部分数据，若要全量可去掉切片，例如: dataset
        # dataset = dataset.select(range(1000)) 
        
        total_len = len(dataset)
        print(f"⚡️ 正在预计算 {total_len} 条问题的 Embedding (CPU Mode)...")
        
        all_questions_raw = dataset['question']
        all_answers_raw = dataset['answer']
        
        # 预计算 Embeddings (Epoch 间复用)
        all_q_embeddings = self.embedder.encode(all_questions_raw, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        print(f"🔥 开始全量进化训练 (vLLM Speedup) - 目标准确率: {TARGET_ACCURACY}%")

        epoch = 0
        best_acc = 0.0

        # [修改逻辑] 只要没达到目标准确率且未超时，就持续训练
        while best_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS:
            epoch += 1
            print(f"\n======== Epoch {epoch}/{MAX_EPOCHS} ========")
            
            # 1. Shuffle 数据索引
            indices = np.random.permutation(total_len)
            
            epoch_correct_count = 0
            epoch_total_count = 0
            
            # 使用 tqdm 显示当前 Epoch 进度
            pbar = tqdm(range(0, total_len, CHUNK_SIZE), desc=f"Epoch {epoch} Training")
            
            for chunk_start in pbar:
                chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
                current_batch_indices = indices[chunk_start:chunk_end]
                
                # 根据打乱的索引获取数据
                chunk_questions = [all_questions_raw[i] for i in current_batch_indices]
                chunk_answers = [all_answers_raw[i] for i in current_batch_indices]
                # 获取对应的预计算 embedding
                chunk_q_embeddings = all_q_embeddings[current_batch_indices]
                
                # --- A. 批量检索 ---
                retrieved_batch = self.memory.batch_retrieve(chunk_q_embeddings, top_k=3, threshold=0.4)
                
                used_rag_data = []      # 用于 update_scores (存放 List[tuple])
                inference_prompts = []  # 用于 vLLM 推理
                
                for idx, q in enumerate(chunk_questions):
                    prompt = self.construct_prompt(q)
                    inference_prompts.append(prompt)
                    used_rag_data.append([])
                
                # --- B. vLLM 批量推理 ---
                model_outputs = self.batch_generate_vllm(inference_prompts, self.params_inference)
                
                # --- C. 评估 & 准备反馈 ---
                is_correct_list = []
                correct_samples = [] 
                incorrect_indices = []
                
                for idx, pred in enumerate(model_outputs):
                    gt = chunk_answers[idx]
                    is_right = self.check_answer(pred, gt)
                    is_correct_list.append(is_right)
                    
                    if is_right:
                        correct_samples.append((chunk_questions[idx], gt)) 
                        epoch_correct_count += 1
                    else:
                        incorrect_indices.append(idx)
                    epoch_total_count += 1
                
                # 更新分数
                self.memory.update_scores_batch(used_rag_data, is_correct_list, model_outputs)

                with open(self.rag_log_path, "a") as f:
                    for i in range(len(chunk_questions)):
                        log_entry = {
                            "epoch": epoch,
                            "question": chunk_questions[i],
                            "is_correct": is_correct_list[i],
                            "used_rules": [
                                {
                                    "sid": sid,
                                    "distance": float(dist),
                                    "content": content
                                }
                                for (content, dist, sid) in used_rag_data[i]
                            ]
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # --- D. 批量反思与自生成经验 (只针对错题) ---
                if incorrect_indices:
                    reflect_prompts = []
                    verify_data = [] 
                    
                    for idx in incorrect_indices:
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
                            orig_gt = verify_data[temp_candidates[m][2]][1]
                            if self.check_answer(pred, orig_gt):
                                verified_patterns.append(temp_candidates[m][0])
                                verified_strategies.append(temp_candidates[m][1])
                        
                        if verified_patterns:
                            p_embeds = self.embedder.encode(verified_patterns, convert_to_numpy=True).tolist()
                            self.memory.add_experience_batch(verified_patterns, verified_strategies, p_embeds)

                # --- E. 定期淘汰 ---
                if (chunk_start // CHUNK_SIZE) % 5 == 0:
                    self.memory.prune_db(threshold=0.25)

                # 更新进度条信息
                batch_acc = len(correct_samples) / len(chunk_questions) * 100
                pbar.set_postfix({"Batch Acc": f"{batch_acc:.1f}%", "DB Size": self.memory.collection.count()})
            
            # --- Epoch 总结 ---
            current_epoch_acc = (epoch_correct_count / epoch_total_count) * 100
            print(f"\n📊 Epoch {epoch} 完成 | 准确率: {current_epoch_acc:.2f}% (Target: {TARGET_ACCURACY}%)")
            
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
        """
        显式释放 vLLM 资源和显存
        """
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