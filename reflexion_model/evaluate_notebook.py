from config_eval import * 
import os
import re
import time
import torch

import chromadb
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= 1. 记忆管理器 (保持不变) =================
class MemoryManager:
    def __init__(self):
        self.DB_PATH = DB_PATH
        self.client = chromadb.PersistentClient(path=self.DB_PATH)
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

# ================= 2. 科学对比评估器 (完整修改版) =================
class ScientificComparator:
    def __init__(self):
        print(f"🚀 初始化 vLLM 引擎 (Rigorous Mode)...")
        self.MODEL_PATH = MODEL_NAME
        self.DB_PATH = DB_PATH
        self.GPU_UTILIZATION = GPU_MEMORY_UTILIZATION
        self.TOP_K = TOP_K
        self.SC_PATHS = SC_PATHS 
        # [修改] 收紧阈值，只有非常匹配的规则才启用 RAG，防止噪音干扰
        self.RAG_THRESHOLD = RAG_THRESHOLD  # 建议设为 0.35 或 0.4，越小越严
        
        self.llm = LLM(
            model=self.MODEL_PATH, 
            trust_remote_code=True,
            gpu_memory_utilization=self.GPU_UTILIZATION,
            tensor_parallel_size=1, 
            max_model_len=2048
        )
        
        # 1. 解题用的采样参数
        self.params_sc = SamplingParams(
            n=self.SC_PATHS, 
            temperature=0.3, 
            top_p=0.9, 
            max_tokens=1024,
            stop=["<|endoftext|>", "<|im_end|>", "Question:"]
        )

        # 2. 抽象意图用的采样参数
        self.params_greedy = SamplingParams(
            temperature=0.0, 
            max_tokens=128,
            stop=["<|endoftext|>", "<|im_end|>", "\n\n"]
        )

        print("📥 加载 Embedder (CPU)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.memory = MemoryManager()

    def construct_base_prompt(self, question):
        # 标准 CoT Prompt
        return f"<|im_start|>user\nQuestion: {question}\nLet's think step by step.\nAnswer:<|im_end|>\n<|im_start|>assistant\n"

    def construct_rag_prompt(self, question, retrieved_items):
        valid_items = [item[0] for item in retrieved_items if item[1] < self.RAG_THRESHOLD]
        
        # [逻辑保护] 如果过滤后没有规则，返回 None，指示调用者使用 Base 结果
        if not valid_items:
            return None
        
        context_str = "\n".join([f"[Rule {i+1}]: {rule}" for i, rule in enumerate(valid_items)])
        
        # [关键修改] 
        # 1. 语气变软：Reference Rules (Only if helpful)
        # 2. 核心修复：句尾加回 "Let's think step by step." 激活模型智商
        prompt = f"""<|im_start|>user
[Reference Rules (Use ONLY if helpful)]
{context_str}

[Question]
{question}

Let's think step by step using the rules above if applicable.
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

    def batch_abstract_for_retrieval(self, questions):
        # ... (保持原来的抽象函数不变) ...
        prompts = []
        for q in questions:
            content = f"""Task: Identify the core mathematical concept. Output 1 abstract sentence without numbers.
[Example] Q: John has 5 apples... A: Calculating total sum.
[Target] Q: {q}
A:"""
            prompts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        
        print("   🧠 Abstracting questions...")
        outputs = self.llm.generate(prompts, self.params_greedy, use_tqdm=True)
        return [output.outputs[0].text.strip() for output in outputs]

    # ================= [核心修改] 评估主循环 =================
    def run_scientific_test(self):
        dataset = load_dataset("gsm8k", "main")['test']
        questions = dataset['question']
        ground_truths = dataset['answer']
        total = len(questions)
        
        print(f"📊 测试集大小: {total} | SC={self.SC_PATHS} | RAG Threshold={self.RAG_THRESHOLD}")

        # ------------------------------------------------------------------
        # Phase 1: Base Model (保存 Base 结果用于回退！)
        # ------------------------------------------------------------------
        print(f"\n🔵 [Group A] Base Model...")
        base_prompts = [self.construct_base_prompt(q) for q in questions]
        
        # 运行 Base 推理
        t0 = time.time()
        base_outputs_obj = self.llm.generate(base_prompts, self.params_sc, use_tqdm=True)
        print(f"   Base 耗时: {time.time()-t0:.2f}s")

        # 计算并缓存 Base 的结果
        base_predictions = []
        correct_base = 0
        for i, out in enumerate(base_outputs_obj):
            pred = self.majority_vote(out)
            base_predictions.append(pred) # 存起来！
            if self.check_correct(pred, ground_truths[i]):
                correct_base += 1
        
        acc_base = correct_base / total * 100
        print(f"   ✅ Base Accuracy: {acc_base:.2f}%")

        # ------------------------------------------------------------------
        # Phase 2: Hybrid RAG Model (Fallback Logic)
        # ------------------------------------------------------------------
        print(f"\n🟢 [Group B] Hybrid RAG (Recall & Fallback)...")
        print("   策略：仅当检索到的经验极其匹配时(Distance < Threshold)才启用 RAG，否则直接复用 Base 答案。")
        
        # 1. 抽象 + 检索
        abstract_queries = self.batch_abstract_for_retrieval(questions)
        print("   -> Encoding & Retrieving...")
        abstract_embeddings = self.embedder.encode(abstract_queries, batch_size=64, convert_to_numpy=True).tolist()
        all_retrieved = self.memory.batch_retrieve(abstract_embeddings, top_k=self.TOP_K)
        
        # 2. 构建混合任务列表
        rag_prompts = []
        rag_indices = [] # 记录哪些题目需要跑 RAG
        final_rag_preds = [None] * total # 预填充列表
        
        skipped_count = 0
        
        for i, q in enumerate(questions):
            # 尝试构建 RAG prompt，如果距离太远，construct_rag_prompt 会返回 None
            prompt = self.construct_rag_prompt(q, all_retrieved[i])
            
            if prompt is None:
                # [回退逻辑]：经验不可靠，直接复用 Base 的预测结果！
                # 这样可以保证准确率绝对不会因为“强行RAG”而低于 Base (除非 RAG 把原本对的改错了)
                final_rag_preds[i] = base_predictions[i]
                skipped_count += 1
            else:
                # 经验可靠，加入重算队列
                rag_prompts.append(prompt)
                rag_indices.append(i)

        print(f"   ℹ️  RAG 触发率: {len(rag_indices)}/{total} (Fallback to Base: {skipped_count})")

        # 3. 只对触发了 RAG 的题目进行推理 (节省大量时间！)
        if rag_prompts:
            print(f"   🚀 Running RAG Inference on {len(rag_prompts)} samples...")
            t0 = time.time()
            rag_inference_outputs = self.llm.generate(rag_prompts, self.params_sc, use_tqdm=True)
            print(f"   RAG 部分耗时: {time.time()-t0:.2f}s")
            
            # 填回结果
            for idx_in_batch, out in enumerate(rag_inference_outputs):
                original_idx = rag_indices[idx_in_batch]
                pred = self.majority_vote(out)
                final_rag_preds[original_idx] = pred
        
        # 4. 统计 Group B 最终结果
        correct_rag = 0
        for i, pred in enumerate(final_rag_preds):
            if self.check_correct(pred, ground_truths[i]):
                correct_rag += 1
        
        acc_rag = correct_rag / total * 100
        print(f"   ✅ Hybrid RAG Accuracy: {acc_rag:.2f}%")

        # ================= 最终分析 =================
        print("\n" + "="*60)
        print("🧪 最终报告")
        print("="*60)
        print(f"1. Base Model Acc  : {acc_base:.2f}%")
        print(f"2. Hybrid RAG Acc  : {acc_rag:.2f}%")
        print("-" * 60)
        diff = acc_rag - acc_base
        print(f"📈 净提升: {diff:+.2f}%")
        
        if diff >= 0:
            print("结论：混合策略生效。系统保留了基座能力，并在有经验时获得了增益。")
        else:
            print("结论：仍然有下降？请检查 RAG Prompt 是否干扰了模型。")
        print("="*60)


if __name__ == "__main__":

    
    try:
        evaluator = ScientificComparator()
        evaluator.run_scientific_test()
    except KeyboardInterrupt:
        print("\n🛑 评估被用户中断")
    except Exception as e:
        print(f"\n❌ 评估出错: {e}")