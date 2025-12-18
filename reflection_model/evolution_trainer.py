import os
import shutil
import time
import random
import re
import torch
import chromadb
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ================= 配置区域 =================
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DB_PATH = "./reflexion_full_db"
CHUNK_SIZE = 64         
INFERENCE_BATCH_SIZE = 32 
TRAIN_BATCH_SIZE = 8    
LEARNING_RATE = 2e-5    
MAX_NEW_TOKENS = 256

# ================= 1. 记忆管理器 (修复版) =================
class MemoryManager:
    def __init__(self, reset=True):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        # 内存缓存统计数据
        self.skill_stats = {} 

    def batch_retrieve(self, query_embeddings, n_results=1):
        """批量检索"""
        count = self.collection.count()
        if count == 0: return [None] * len(query_embeddings)
        
        results_list = []
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=n_results)
            for i in range(len(query_embeddings)):
                if results['documents'][i]:
                    # 返回 (Content, Distance, ID)
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
        """批量写入"""
        if not patterns_A: return
        # 生成唯一ID
        new_ids = [f"rule_{int(time.time())}_{i}_{random.randint(0,999)}" for i in range(len(patterns_A))]
        metadatas = [{"pattern": p} for p in patterns_A]
        
        self.collection.add(
            ids=new_ids,
            embeddings=embeddings_A,
            documents=strategies_B,
            metadatas=metadatas
        )
        # 初始化分数
        for sid in new_ids:
            self.skill_stats[sid] = {"score": 0.5, "usage": 0, "history_correct": 0}

    def update_scores_batch(self, usage_data, is_correct_list, model_outputs):
        """
        【修复】更新分数逻辑
        :param usage_data: List of (skill_id, skill_content) or None
        :param is_correct_list: List of Boolean
        :param model_outputs: List of String (模型的回答)
        """
        for i, item in enumerate(usage_data):
            if item is None: continue # 没用RAG
            
            sid, content = item
            is_correct = is_correct_list[i]
            output_text = model_outputs[i]
            
            # 初始化 stats
            if sid not in self.skill_stats:
                self.skill_stats[sid] = {"score": 0.5, "usage": 0, "history_correct": 0}
            stats = self.skill_stats[sid]
            
            # --- 策略：无辜旁观者保护 ---
            # 如果经验的内容（Trigger/Strategy）完全没有出现在模型的思考中，说明模型可能忽略了它
            # 这种情况下，不应该因为做错了而惩罚经验
            # (简单的关键词匹配，取前20个字符作为指纹)
            if len(content) > 10:
                fingerprint = content[:10]
                if fingerprint not in output_text and not is_correct:
                    continue # 没用上，且做错了 -> 不怪经验，跳过更新

            stats['usage'] += 1
            
            if is_correct:
                stats['history_correct'] += 1
                # 奖励
                stats['score'] = min(1.0, stats['score'] + 0.1)
            else:
                # 惩罚
                penalty = 0.2
                # 老兵免疫：如果历史战绩好，惩罚减轻
                if stats['history_correct'] > 10: penalty = 0.1
                if stats['history_correct'] > 50: penalty = 0.05
                
                stats['score'] = max(0.0, stats['score'] - penalty)

    def prune_db(self, min_usage=5, threshold=0.2):
        """淘汰逻辑"""
        ids_to_delete = []
        for sid, stats in list(self.skill_stats.items()):
            if stats['usage'] >= min_usage and stats['score'] < threshold:
                ids_to_delete.append(sid)
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.skill_stats[sid]
            return len(ids_to_delete)
        return 0

# ================= 2. 全量进化训练器 (修复调用) =================
class ReflexionTrainerFull:
    def __init__(self):
        print("🚀 初始化训练环境 (Full Set Mode)...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side="left")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
        )
        
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.model = get_peft_model(self.base_model, peft_config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LEARNING_RATE)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cuda" if torch.cuda.is_available() else "cpu")
        self.memory = MemoryManager(reset=True)

    def batch_generate(self, prompts, temperature=0.5):
        self.model.eval()
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, temperature=temperature, do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def train_on_chunk(self, training_data):
        if not training_data: return 0.0
        self.model.train()
        total_loss = 0; steps = 0
        self.tokenizer.padding_side = "right"
        
        for i in range(0, len(training_data), TRAIN_BATCH_SIZE):
            batch = training_data[i : i + TRAIN_BATCH_SIZE]
            texts = [f"Question: {q}\nAnswer: {a}{self.tokenizer.eos_token}" for q, a in batch]
            inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
            outputs = self.model(**inputs, labels=inputs.input_ids)
            loss = outputs.loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss.item(); steps += 1
            
        self.tokenizer.padding_side = "left"
        return total_loss / max(1, steps)

    def parse_reflection(self, texts):
        patterns = []; strategies = []
        for text in texts:
            try:
                if "**Trigger (A)**:" in text and "**Strategy (B)**:" in text:
                    parts = text.split("**Strategy (B)**:")
                    p = parts[0].replace("**Trigger (A)**:", "").strip()
                    s = parts[1].strip()
                    if len(p) > 5 and len(s) > 5:
                        patterns.append(p); strategies.append(s)
            except: continue
        return patterns, strategies

    def run_full_evolution(self):
        dataset = load_dataset("gsm8k", "main")['train']
        total_len = len(dataset)
        print(f"🔥 开始全量进化训练，数据总量: {total_len}")

        for chunk_start in range(0, total_len, CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
            chunk_data = dataset.select(range(chunk_start, chunk_end))
            chunk_questions = chunk_data['question']
            chunk_answers = chunk_data['answer']
            
            # --- 1. 批量检索 ---
            q_embeds = self.embedder.encode(chunk_questions).tolist()
            retrieved_results = self.memory.batch_retrieve(q_embeds)
            
            # --- 2. 构造 Prompts & 记录用到的 Skill ---
            inference_prompts = []
            used_rag_data = [] # 记录 [(sid, content), None, ...] 用于更新分数
            
            for idx, q in enumerate(chunk_questions):
                res = retrieved_results[idx]
                if res and res[1] < 1.0: # 距离阈值
                    content, dist, sid = res
                    # 只有分数不算太烂的才用
                    curr_score = self.memory.skill_stats.get(sid, {}).get("score", 0.5)
                    if curr_score > 0.2:
                        prompt = f"Hint: {content}\nQuestion: {q}\nAnswer step-by-step:"
                        used_rag_data.append((sid, content))
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
                batch_outs = self.batch_generate(batch_prompts)
                model_outputs.extend(batch_outs)
            
            # --- 4. 评估 ---
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
            
            # >>> 核心修改：传入 model_outputs 以支持旁观者保护 <<<
            self.memory.update_scores_batch(used_rag_data, is_correct_list, model_outputs)

            # --- 5. 批量反思 ---
            if incorrect_indices:
                reflect_prompts = []
                # 记录原始问题和答案，用于验证
                verify_data = [] 
                
                for idx in incorrect_indices:
                    q = chunk_questions[idx]
                    gt = chunk_answers[idx]
                    p = f"""
I failed this problem.
Problem: {q}
Solution: {gt}
Summarize a general math rule to solve this.
Format:
**Trigger (A)**: [Short Pattern]
**Strategy (B)**: [Short Logic]
"""
                    reflect_prompts.append(p)
                    verify_data.append((q, gt))
                
                # A. 批量生成反思
                reflections = self.batch_generate(reflect_prompts, temperature=0.7)
                
                # B. 解析反思结果
                parsed_patterns, parsed_strategies, valid_indices = [], [], []
                
                # 解析出 pattern 和 strategy
                temp_candidates = []
                for k, text in enumerate(reflections):
                    try:
                        if "**Trigger (A)**:" in text and "**Strategy (B)**:" in text:
                            parts = text.split("**Strategy (B)**:")
                            p_text = parts[0].replace("**Trigger (A)**:", "").strip()
                            s_text = parts[1].strip()
                            if len(p_text) > 5 and len(s_text) > 5:
                                temp_candidates.append((p_text, s_text, k))
                    except:
                        continue
                
                # C. >>> 核心修改：验证环节 (Verification) <<<
                # 拿刚才生成的“经验”，立刻去试着解一遍原题
                if temp_candidates:
                    verify_prompts = []
                    for p_text, s_text, k in temp_candidates:
                        orig_q = verify_data[k][0]
                        # 强制模型使用新生成的经验解题
                        vp = f"Hint: {s_text}\nQuestion: {orig_q}\nAnswer step-by-step:"
                        verify_prompts.append(vp)
                    
                    # 批量验证推理
                    verify_outputs = self.batch_generate(verify_prompts, temperature=0.1)
                    
                    # 只有做对的，才存入数据库！
                    verified_patterns = []
                    verified_strategies = []
                    
                    for m, pred in enumerate(verify_outputs):
                        orig_gt = verify_data[temp_candidates[m][2]][1]
                        if self.check_answer(pred, orig_gt):
                            # 🎉 验证通过！这条经验是有用的！
                            verified_patterns.append(temp_candidates[m][0])
                            verified_strategies.append(temp_candidates[m][1])
                    
                    # D. 存入经过验证的高质量经验
                    if verified_patterns:
                        p_embeds = self.embedder.encode(verified_patterns).tolist()
                        self.memory.add_experience_batch(verified_patterns, verified_strategies, p_embeds)
                        print(f"✨ [验证通过] 新增 {len(verified_patterns)} 条有效经验 (淘汰了 {len(temp_candidates) - len(verified_patterns)} 条垃圾经验)")
                    else:
                        print(f"💀 [验证失败] 生成的 {len(temp_candidates)} 条经验全是无效的")
                        
            # --- 6. 批量微调 ---
            loss = 0
            if correct_samples:
                loss = self.train_on_chunk(correct_samples)

            # --- 7. 定期淘汰 ---
            pruned_count = 0
            if (chunk_start // CHUNK_SIZE) % 5 == 0:
                pruned_count = self.memory.prune_db(min_usage=5, threshold=0.25)

            acc = len(correct_samples) / len(chunk_data) * 100
            print(f"Chunk {chunk_start//CHUNK_SIZE}: Acc={acc:.1f}% | Pruned={pruned_count} | DB Size={self.memory.collection.count()}")

            if (chunk_start // CHUNK_SIZE) % 5 == 0:
                self.model.save_pretrained("./evolved_qwen_lora_checkpoint")

        print("全量训练完成！")
        self.model.save_pretrained("./evolved_qwen_lora")

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