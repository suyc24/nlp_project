import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import shutil
import time
import random
import re
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DB_PATH = "./reflexion_full_db"
CHUNK_SIZE = 64         
INFERENCE_BATCH_SIZE = 32 
MAX_NEW_TOKENS = 256

# ================= 1. 记忆管理器 (保持不变) =================
class MemoryManager:
    def __init__(self, reset=True):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        self.skill_stats = {} 
        self.current_step = 0 

    def batch_retrieve(self, query_embeddings, n_results=1):
        count = self.collection.count()
        if count == 0: return [None] * len(query_embeddings)
        
        results_list = []
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=n_results)
            for i in range(len(query_embeddings)):
                if results['documents'][i]:
                    doc = results['documents'][i][0]
                    dist = results['distances'][i][0]
                    sid = results['ids'][i][0]
                    results_list.append((doc, dist, sid))
                else:
                    results_list.append(None)
        except:
            return [None] * len(query_embeddings)
            
        return results_list

    def add_experience_batch(self, patterns_A, strategies_B, embeddings_A):
        if not patterns_A: return
        
        unique_patterns = []
        unique_strategies = []
        unique_embeddings = []
        
        for i, emb in enumerate(embeddings_A):
            try:
                # 简单的去重逻辑
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
                "score": 0.5,       
                "usage": 0,         
                "history_correct": 0, 
                "created_step": self.current_step,
                "is_probation": True 
            }

    def update_scores_batch(self, usage_data, is_correct_list, model_outputs):
        self.current_step += 1 
        
        for i, item in enumerate(usage_data):
            if item is None: continue 
            
            sid, _, content = item
            is_correct = is_correct_list[i]
            output_text = model_outputs[i]
            
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
            if age > 500 and stats['usage'] < 2:
                ids_to_delete.append(sid)

        if ids_to_delete:
            print(f"✂️ [强力淘汰] 清理了 {len(ids_to_delete)} 条低质经验 (当前剩余: {len(self.skill_stats) - len(ids_to_delete)})")
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.skill_stats[sid]
            return len(ids_to_delete)
        return 0

# ================= 2. 全量进化训练器 =================
class ReflexionTrainerFull:
    def __init__(self):
        print("🚀 初始化训练环境 (Self-Evolving Mode)...")
        # 尝试使用 flash_attention_2 或 sdpa，如果报错会自动回退
        self.attn_impl = "sdpa" if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else "eager"
        print(f"   -> Attention Implementation: {self.attn_impl}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left", cache_dir=HF_CACHE_DIR)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            dtype=torch.float16,
            attn_implementation=self.attn_impl,
            cache_dir=HF_CACHE_DIR,
            low_cpu_mem_usage=True
        )
        self.model = self.base_model
        self.model.eval()
        
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu", cache_folder=HF_CACHE_DIR)
        self.memory = MemoryManager(reset=True)

    def batch_generate(self, prompts, temperature=0.5, max_new_tokens=256):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens, 
                temperature=temperature, 
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    # --- 辅助函数：检测生成的规则是否包含具体数字 ---
    def has_specific_numbers(self, text, question_text):
        """
        防止模型把题目的具体数字写进规则里。
        如果规则里出现了题目中大于10的数字，或者复杂的浮点数，大概率是过拟合了。
        """
        # 提取题目中的数字
        q_nums = set(re.findall(r'\b\d+\b', question_text))
        # 提取规则中的数字
        r_nums = set(re.findall(r'\b\d+\b', text))
        
        # 允许的小数字白名单 (规则中常见的系数)
        whitelist = {'1', '2', '3', '4', '10', '100', '180', '360'} 
        
        for n in r_nums:
            if n in q_nums and n not in whitelist:
                return True # 发现了题目特定的数字，判定为劣质规则
        return False

    def run_full_evolution(self):
        dataset = load_dataset("gsm8k", "main")['train']
        total_len = len(dataset)
        
        print(f"⚡️ 正在预计算 {total_len} 条问题的 Embedding...")
        all_questions = dataset['question']
        all_q_embeddings = self.embedder.encode(all_questions, batch_size=256, show_progress_bar=True).tolist()
        
        print(f"🔥 开始全量进化训练 (模型自生成经验)...")

        for chunk_start in range(0, total_len, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
            chunk_indices = range(chunk_start, chunk_end)
            
            chunk_questions = [dataset[i]['question'] for i in chunk_indices]
            chunk_answers = [dataset[i]['answer'] for i in chunk_indices]
            
            # --- 1. 批量检索 ---
            chunk_q_embeddings = all_q_embeddings[chunk_start:chunk_end]
            retrieved_results = self.memory.batch_retrieve(chunk_q_embeddings)
            
            # --- 2. 构造 Inference Prompts ---
            inference_prompts = []
            used_rag_data = [] 
            
            for idx, q in enumerate(chunk_questions):
                res = retrieved_results[idx]
                if res and res[1] < 0.6: # 收紧阈值，防止无关干扰
                    content, dist, sid = res
                    curr_score = self.memory.skill_stats.get(sid, {}).get("score", 0.5)
                    
                    if curr_score > 0.3:
                        # 使用更柔和的提示词
                        prompt = f"Reference rule: {content}\nQuestion: {q}\nAnswer step-by-step:"
                        used_rag_data.append((sid, dist, content)) 
                    else:
                        prompt = f"Question: {q}\nAnswer step-by-step:"
                        used_rag_data.append(None)
                else:
                    prompt = f"Question: {q}\nAnswer step-by-step:"
                    used_rag_data.append(None)
                inference_prompts.append(prompt)
            
            # --- 3. 批量推理 ---
            model_outputs = []
            for i in range(0, len(inference_prompts), INFERENCE_BATCH_SIZE):
                batch_prompts = inference_prompts[i : i + INFERENCE_BATCH_SIZE]
                batch_outs = self.batch_generate(batch_prompts, max_new_tokens=MAX_NEW_TOKENS)
                model_outputs.extend(batch_outs)
            
            # --- 4. 评估 & 反馈更新 ---
            is_correct_list = []
            correct_samples = [] 
            incorrect_indices = []
            
            for idx, pred in enumerate(model_outputs):
                gt = chunk_answers[idx]
                is_right = self.check_answer(pred, gt)
                is_correct_list.append(is_right)
                
                if is_right:
                    correct_samples.append((chunk_questions[idx], gt)) 
                else:
                    incorrect_indices.append(idx)
            
            self.memory.update_scores_batch(used_rag_data, is_correct_list, model_outputs)

            # --- 5. 批量反思与自生成经验 (核心修改点) ---
            if incorrect_indices:
                reflect_prompts = []
                verify_data = [] 
                
                for idx in incorrect_indices:
                    q = chunk_questions[idx]
                    gt = chunk_answers[idx]
                    
                    # === 核心修改：Teach-Model Prompt ===
                    # 给定一个“单样本(One-shot)”，教 0.5B 模型如何做抽象总结
                    # 明确要求使用变量 (X, Y) 而不是数字
                    p = f"""
Task: Extract a general math rule from the incorrect problem.
INSTRUCTION: 
1. Ignore specific names (like "John") and numbers (like "50").
2. Describe the LOGIC used to solve it.
3. Use variables like X, Y, Z.

[EXAMPLE]
Problem: John buys 5 apples for $2 each. Total cost?
Solution: 5 * 2 = 10.
**Trigger (A)**: Calculating total cost from quantity and rate.
**Strategy (B)**: Total Cost = Quantity * Unit Price.

[YOUR TURN]
Problem: {q}
Solution: {gt}

Summarize the rule:
**Trigger (A)**: [General pattern]
**Strategy (B)**: [General logic using variables]
"""
                    reflect_prompts.append(p)
                    verify_data.append((q, gt))
                
                # 稍微调高 Temperature 让它有一点创造力去概括
                reflections = self.batch_generate(reflect_prompts, temperature=0.6, max_new_tokens=128)
                
                temp_candidates = []
                for k, text in enumerate(reflections):
                    try:
                        if "**Trigger (A)**:" in text and "**Strategy (B)**:" in text:
                            parts = text.split("**Strategy (B)**:")
                            p_text = parts[0].replace("**Trigger (A)**:", "").strip()
                            s_text = parts[1].strip()
                            
                            # 过滤掉包含题目特定数字的“伪规则”
                            orig_q = verify_data[k][0]
                            if self.has_specific_numbers(s_text, orig_q):
                                continue # 跳过这条过拟合的规则

                            if len(p_text) > 5 and len(s_text) > 5:
                                temp_candidates.append((p_text, s_text, k))
                    except: continue
                
                # C. 验证环节 (Verificaton Loop)
                if temp_candidates:
                    verify_prompts = []
                    for p_text, s_text, k in temp_candidates:
                        orig_q = verify_data[k][0]
                        # 验证时，强制模型应用这条新生成的规则
                        vp = f"Rule: {s_text}\nQuestion: {orig_q}\nAnswer step-by-step:"
                        verify_prompts.append(vp)
                    
                    verify_outputs = self.batch_generate(verify_prompts, temperature=0.1, max_new_tokens=256)
                    
                    verified_patterns = []
                    verified_strategies = []
                    
                    for m, pred in enumerate(verify_outputs):
                        orig_gt = verify_data[temp_candidates[m][2]][1]
                        # 只有当这条规则真的帮模型把题做对了，才存入库
                        if self.check_answer(pred, orig_gt):
                            verified_patterns.append(temp_candidates[m][0])
                            verified_strategies.append(temp_candidates[m][1])
                    
                    if verified_patterns:
                        p_embeds = self.embedder.encode(verified_patterns).tolist()
                        self.memory.add_experience_batch(verified_patterns, verified_strategies, p_embeds)
                        print(f"✨ [自生成经验] 新增 {len(verified_patterns)} 条通用规则 (已过滤数字)")

            # --- 6. 定期淘汰 ---
            pruned_count = 0
            if (chunk_start // CHUNK_SIZE) % 5 == 0:
                pruned_count = self.memory.prune_db(threshold=0.25)

            acc = len(correct_samples) / len(chunk_questions) * 100
            db_size = self.memory.collection.count()
            print(f"Chunk {chunk_start//CHUNK_SIZE}: Acc={acc:.1f}% | DB={db_size} | Pruned={pruned_count}")

        print("全量训练完成！")

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

if __name__ == "__main__":
    trainer = ReflexionTrainerFull()
    trainer.run_full_evolution()