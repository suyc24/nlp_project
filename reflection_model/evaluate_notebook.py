import os
import torch
import re
import json
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import chromadb
from peft import PeftModel

# ================= 配置 =================
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_PATH = "./evolved_qwen_lora" # 你的 LoRA 权重路径
DB_PATH = "./reflexion_full_db"   # 你的经验库路径
BATCH_SIZE = 32
OUTPUT_FILE = "final_evaluation_report.json"

# ================= 工具类 =================
class MemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name="rule_book")
        self.stats = {} 
        self._load_cache()

    def _load_cache(self):
        try:
            existing = self.collection.get()
            if existing['ids']:
                for i, sid in enumerate(existing['ids']):
                    self.stats[sid] = existing['metadatas'][i]
        except:
            print("⚠️ 警告：无法加载经验库缓存，可能是新库或路径错误")

    def batch_retrieve(self, query_embeddings):
        count = self.collection.count()
        if count == 0: return [None] * len(query_embeddings)
        
        results_list = []
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=min(5, count))
            for i in range(len(query_embeddings)):
                if results['ids'][i]:
                    # 取 Top-1
                    content = results['documents'][i][0]
                    dist = results['distances'][i][0]
                    sid = results['ids'][i][0]
                    meta = self.stats.get(sid, results['metadatas'][i][0])
                    # 返回内容和距离
                    results_list.append((content, dist))
                else:
                    results_list.append(None)
        except:
            return [None] * len(query_embeddings)
            
        return results_list

# ================= 评估器 =================
class Evaluator:
    def __init__(self):
        print(f"🚀 1. 加载基座模型 (Base Model): {BASE_MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, padding_side="left")
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 先只加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            device_map="auto"
        )
        
        print("📥 加载 Embedder...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device=self.device)
        self.memory = MemoryManager()
        
    def load_lora(self):
        """在基座模型上挂载 LoRA"""
        print(f"\n🧬 2. 挂载进化权重 (LoRA): {LORA_PATH}...")
        # 使用 PeftModel 加载 LoRA，不进行 merge_and_unload 以便对比（或者直接覆盖）
        self.model = PeftModel.from_pretrained(self.model, LORA_PATH)
        self.model.eval()

    def batch_generate(self, prompts):
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512,
                temperature=0.01, # 测试时趋近于 0，消除随机性
                do_sample=False,  # 使用 Greedy Search 保证结果稳定
                pad_token_id=self.tokenizer.pad_token_id
            )
        decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return [d.strip() for d in decoded]

    def extract_answer(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    def check_correct(self, pred_str, ground_truth):
        if "####" in ground_truth:
            gold = self.extract_answer(ground_truth.split("####")[1])
        else:
            gold = self.extract_answer(ground_truth)
        pred = self.extract_answer(pred_str)
        if gold is None or pred is None: return False
        return abs(gold - pred) < 1e-4

    def run_full_comparison(self):
        dataset = load_dataset("gsm8k", "main")['test']
        print(f"\n=== 开始三方对比测试 (Test Set: {len(dataset)}) ===")
        
        total = len(dataset)
        
        # 结果容器
        results_base = []
        results_lora_naked = []
        results_lora_rag = []
        
        questions = dataset['question']
        ground_truths = dataset['answer']
        
        # --- 第一阶段：测试纯基座模型 (Base Model) ---
        print("\n[Phase 1] 测试基座模型 (Base Model)...")
        for i in tqdm(range(0, total, BATCH_SIZE)):
            batch_q = questions[i : i+BATCH_SIZE]
            prompts = [f"Question: {q}\nLet's think step by step.\nAnswer:" for q in batch_q]
            answers = self.batch_generate(prompts)
            results_base.extend(answers)
            
        # --- 第二阶段：加载 LoRA 并测试 ---
        self.load_lora()
        
        # 预先检索所有 RAG 内容 (为了效率)
        print("\n[Retrieval] 正在预检索经验库...")
        all_rag_contexts = []
        for i in tqdm(range(0, total, BATCH_SIZE)):
            batch_q = questions[i : i+BATCH_SIZE]
            q_embeds = self.embedder.encode(batch_q).tolist()
            # 检索
            retrieved = self.memory.batch_retrieve(q_embeds)
            all_rag_contexts.extend(retrieved)

        print("\n[Phase 2 & 3] 测试进化模型 (LoRA & RAG)...")
        for i in tqdm(range(0, total, BATCH_SIZE)):
            batch_q = questions[i : i+BATCH_SIZE]
            
            # A. LoRA Naked Prompts
            prompts_naked = [f"Question: {q}\nLet's think step by step.\nAnswer:" for q in batch_q]
            
            # B. LoRA RAG Prompts
            prompts_rag = []
            for j, q in enumerate(batch_q):
                res = all_rag_contexts[i+j]
                if res:
                    content, dist = res
                    # 如果距离太远，其实不应该用，这里为了强制测试RAG效果，只要有就用
                    p = f"Hint: {content}\nQuestion: {q}\nAnswer step-by-step:"
                else:
                    p = f"Question: {q}\nLet's think step by step.\nAnswer:"
                prompts_rag.append(p)
            
            # 推理
            ans_naked = self.batch_generate(prompts_naked)
            ans_rag = self.batch_generate(prompts_rag)
            
            results_lora_naked.extend(ans_naked)
            results_lora_rag.extend(ans_rag)

        # --- 统计分数 ---
        correct_base = 0
        correct_lora = 0
        correct_rag = 0
        
        for i in range(total):
            gt = ground_truths[i]
            if self.check_correct(results_base[i], gt): correct_base += 1
            if self.check_correct(results_lora_naked[i], gt): correct_lora += 1
            if self.check_correct(results_lora_rag[i], gt): correct_rag += 1
            
        acc_base = correct_base / total * 100
        acc_lora = correct_lora / total * 100
        acc_rag = correct_rag / total * 100
        
        print("\n" + "="*50)
        print("📊 最终三方对比报告")
        print("="*50)
        print(f"1. Base Model (0.5B 原生): {acc_base:.2f}%")
        print(f"2. LoRA Only (内化能力)   : {acc_lora:.2f}%")
        print(f"3. LoRA + RAG (完整能力)  : {acc_rag:.2f}%")
        print("-" * 50)
        print(f"训练带来的内化提升: {acc_lora - acc_base:+.2f}%")
        print(f"RAG带来的额外提升 : {acc_rag - acc_lora:+.2f}%")
        print(f"总提升              : {acc_rag - acc_base:+.2f}%")
        print("="*50)

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.run_full_comparison()