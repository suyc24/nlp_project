import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import re
import torch
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from vllm import LLM, SamplingParams
import time

# ================= 配置 =================
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DB_PATH = "./reflexion_full_db"
GPU_UTILIZATION = 0.9 

# 统一参数：两边都跑 SC
SC_PATHS = 5     
TOP_K = 3        
RAG_THRESHOLD = 0.5 

# ================= 1. 记忆管理器 (不变) =================
class MemoryManager:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_collection(name="rule_book")
        
    def batch_retrieve(self, query_embeddings, top_k=3):
        count = self.collection.count()
        if count == 0: return [[] for _ in range(len(query_embeddings))]
        real_k = min(top_k, count)
        results_list = []
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=real_k)
            for i in range(len(query_embeddings)):
                sample_docs = []
                if results['ids'][i]:
                    for j in range(len(results['ids'][i])):
                        doc = results['documents'][i][j]
                        dist = results['distances'][i][j]
                        sample_docs.append((doc, dist))
                results_list.append(sample_docs)
        except:
            return [[] for _ in range(len(query_embeddings))]
        return results_list

# ================= 2. 科学对比评估器 =================
class ScientificComparator:
    def __init__(self):
        print(f"🚀 初始化 vLLM 引擎 (Rigorous Mode)...")
        
        self.llm = LLM(
            model=MODEL_PATH, 
            trust_remote_code=True,
            gpu_memory_utilization=GPU_UTILIZATION,
            tensor_parallel_size=1, 
            max_model_len=2048
        )
        
        # 【关键】定义统一的采样策略 (SC)
        # 无论是 Base 还是 RAG，都给予 5 次机会进行投票
        self.params_sc = SamplingParams(
            n=SC_PATHS, 
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=256,
            stop=["<|endoftext|>", "<|im_end|>", "Question:"]
        )

        print("📥 加载 Embedder (CPU)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.memory = MemoryManager()

    def construct_base_prompt(self, question):
        return f"<|im_start|>user\nQuestion: {question}\nLet's think step by step.\nAnswer:<|im_end|>\n<|im_start|>assistant\n"

    def construct_rag_prompt(self, question, retrieved_items):
        valid_items = [item[0] for item in retrieved_items if item[1] < RAG_THRESHOLD]
        if not valid_items:
            return self.construct_base_prompt(question)
        
        context_str = "\n".join([f"Rule {i+1}: {rule}" for i, rule in enumerate(valid_items)])
        prompt = f"""<|im_start|>user
You are a math expert. Here are some verified rules that might help solve the problem:
{context_str}

Question: {question}
Instruction: Solve the problem step-by-step. Reference thinking patterns (only if applicable):
Answer:<|im_end|>
<|im_start|>assistant
"""
        return prompt

    def extract_answer(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    def majority_vote(self, request_output):
        """通用投票函数"""
        valid_nums = []
        for output in request_output.outputs:
            num = self.extract_answer(output.text)
            if num is not None:
                valid_nums.append(num)
        if not valid_nums: return None
        return Counter(valid_nums).most_common(1)[0][0]

    def check_correct(self, pred, gt_str):
        if "####" in gt_str:
            gold = self.extract_answer(gt_str.split("####")[1])
        else:
            gold = self.extract_answer(gt_str)
        if pred is None or gold is None: return False
        return abs(pred - gold) < 1e-4

    def run_scientific_test(self):
        dataset = load_dataset("gsm8k", "main")['test']
        questions = dataset['question']
        ground_truths = dataset['answer']
        total = len(questions)
        
        print(f"📊 测试集大小: {total} | 采样路径 n={SC_PATHS} | 控制变量: RAG Context")

        # ================= Phase 1: Base Model (Self-Consistency) =================
        print(f"\n🔵 [Group A] Base Model (Self-Consistency)...")
        base_prompts = [self.construct_base_prompt(q) for q in questions]
        
        t0 = time.time()
        base_outputs = self.llm.generate(base_prompts, self.params_sc, use_tqdm=True)
        print(f"   耗时: {time.time()-t0:.2f}s")

        correct_base = 0
        for i, out in enumerate(base_outputs):
            pred = self.majority_vote(out)
            if self.check_correct(pred, ground_truths[i]):
                correct_base += 1
        
        acc_base = correct_base / total * 100
        print(f"   ✅ Base (SC) Accuracy: {acc_base:.2f}%")

        # ================= Phase 2: RAG Model (Self-Consistency) =================
        print(f"\n🟢 [Group B] RAG Model (Self-Consistency)...")
        
        # 预检索
        print("   -> Retrieving context...")
        q_embeddings = self.embedder.encode(questions, batch_size=64, show_progress_bar=True, convert_to_numpy=True).tolist()
        all_retrieved = self.memory.batch_retrieve(q_embeddings, top_k=TOP_K)
        
        rag_prompts = []
        for i, q in enumerate(questions):
            rag_prompts.append(self.construct_rag_prompt(q, all_retrieved[i]))

        t0 = time.time()
        rag_outputs = self.llm.generate(rag_prompts, self.params_sc, use_tqdm=True)
        print(f"   耗时: {time.time()-t0:.2f}s")

        correct_rag = 0
        for i, out in enumerate(rag_outputs):
            pred = self.majority_vote(out)
            if self.check_correct(pred, ground_truths[i]):
                correct_rag += 1
        
        acc_rag = correct_rag / total * 100
        print(f"   ✅ RAG (SC) Accuracy: {acc_rag:.2f}%")

        # ================= 最终分析 =================
        print("\n" + "="*60)
        print("🧪 科学归因分析 (Ablation Study)")
        print("="*60)
        print(f"控制变量：Self-Consistency (n={SC_PATHS})")
        print("-" * 60)
        print(f"1. 基准能力 (Base + SC)    : {acc_base:.2f}%")
        print(f"2. 进化能力 (Base + SC + RAG): {acc_rag:.2f}%")
        print("-" * 60)
        diff = acc_rag - acc_base
        print(f"📈 知识库净贡献 (Pure RAG Gain): {diff:+.2f}%")
        
        if diff > 0:
            print("结论：RAG 提供了有效的信息增益，不仅仅是引入了随机性。")
        else:
            print("结论：RAG 未能带来正向收益，可能是检索噪音过大或模型未能有效利用提示。")
        print("="*60)

if __name__ == "__main__":
    evaluator = ScientificComparator()
    evaluator.run_scientific_test()