import os
import re
import torch
import numpy as np
import time
import argparse
import yaml
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# 确保缓存路径正确
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR

# 复用 Principle Manager 以确保检索逻辑一致
from principle_manager import MemoryManager

class ScientificComparator:
    def __init__(self, config):
        print(f"🚀 初始化评估引擎 (Adaptive RAG Mode)...")
        self.config = config
        self.MODEL_PATH = config["MODEL_PATH"]
        self.GPU_UTILIZATION = config["GPU_UTILIZATION"]
        self.TOP_K = config["TOP_K"]
        
        # vLLM 初始化
        self.llm = LLM(
            model=self.MODEL_PATH, 
            trust_remote_code=True,
            gpu_memory_utilization=self.GPU_UTILIZATION,
            tensor_parallel_size=1, 
            max_model_len=2048,
            download_dir=HF_CACHE_DIR
        )
        
        # 1. 抽象参数 (Greedy Decode - 必须与训练时完全一致)
        self.params_abstract = SamplingParams(
            temperature=0.0, 
            top_p=0.9, 
            max_tokens=128,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        # 2. 生成参数 (High Temp + Majority Vote)
        # 采样 3 次以计算一致性
        self.params_generate = SamplingParams(
            n=3, 
            temperature=0.2, 
            top_p=0.95, 
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # 3. Baseline 生成参数 (n=6)
        self.params_generate_baseline = SamplingParams(
            n=6, 
            temperature=0.2, 
            top_p=0.95, 
            max_tokens=1024,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        print("📥 加载 Embedder (CPU)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu", cache_folder=HF_CACHE_DIR)
        
        # 加载训练好的记忆库
        self.memory = MemoryManager(reset=False)

    def construct_abstraction_prompt(self, q):
        """
        [关键] 必须与 evolution_trainer.py 完全一致，保证 Trigger 相同
        """
        content = f"""
            Task: Identify the core mathematical concept and intent of the following problem.
            Output a concise, abstract description of the problem and its condition.
            ### Requirements
            - **Format**: Your output must be a single sentence following this pattern: "[Abstract Problem Type] given that [Specific Conditions from the Question including numerical constraints, relationships, and constraints]"
            - **Strict Constraint**: Do NOT include any specific numbers (e.g., 16, 3) or specific nouns (e.g., eggs, ducks) from the current problem. The principle must be universal. 

            [Example]
            Q: John has 5 apples and buys 3 more. How many?
            A: Calculating the total sum of objects given that each part is provided.

            [Target]
            Q: {q}
            A:"""
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def construct_prompt(self, q, context=None):
        """
        标准推理 Prompt
        """
        if context:
            content = f"""
[Reference Rules]
{context}

[Question]
{q}

[Instruction]
1. Read the Reference Rules carefully.
2. First, decide which rule is relevant to the question.
3. If a rule is relevant, write "Selected Rule: [Rule Content]".
4. If no rule is relevant, write "No suitable rule found".
5. Then, solve the problem step-by-step using the selected rule (if any).

Answer:"""
        else:
            content = f"Question: {q}\nAnswer step-by-step:"
            
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def extract_answer(self, text):
        if not text: return None
        # 移除逗号以便解析数字 (e.g., 1,000 -> 1000)
        text = text.replace(',', '')
        # 提取最后一个数字
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    def get_majority_vote(self, request_output):
        """
        返回 (众数答案, 众数出现的次数, 所有有效答案列表)
        """
        valid_nums = []
        for output in request_output.outputs:
            num = self.extract_answer(output.text)
            if num is not None:
                valid_nums.append(num)
        
        if not valid_nums: return None, 0, []
        
        # 找到出现次数最多的答案
        counter = Counter(valid_nums)
        most_common = counter.most_common(1)[0] # (value, count)
        return most_common[0], most_common[1], valid_nums

    def check_correct(self, pred, gt_str):
        if "####" in gt_str:
            gold = self.extract_answer(gt_str.split("####")[1])
        else:
            gold = self.extract_answer(gt_str)
        
        if pred is None or gold is None: return False
        return abs(pred - gold) < 1e-4

    def batch_generate_vllm(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=True)
        return outputs

    def run_scientific_test(self):
        # 加载测试集
        dataset = load_dataset("gsm8k", "main")['test']
        # 为了快速演示，这里可以切片，全量测试请去掉切片
        # dataset = dataset.select(range(200)) 
        questions = dataset['question']
        ground_truths = dataset['answer']
        total = len(questions)
        
        print(f"📊 Test Set Size: {total}")
        print(f"⚙️  Settings: Temp=1.0, N=3 (Majority Vote)")

        # ======================================================================
        # Phase 1: Baseline (No RAG) - 6次采样 (公平对比)
        # ======================================================================
        print(f"\n🔵 [Group A] Baseline (No RAG, n=6)...")
        base_prompts = [self.construct_prompt(q, context=None) for q in questions]
        base_outputs = self.batch_generate_vllm(base_prompts, self.params_generate_baseline)

        # ======================================================================
        # Phase 2: RAG (With Abstraction) - 3次采样 (用于 Adaptive 组合)
        # ======================================================================
        print(f"\n🟢 [Group B] RAG (With Abstraction)...")
        
        # 1. Abstract (Greedy)
        print("   🧠 Step 1: Abstracting questions (Greedy)...")
        abstract_prompts = [self.construct_abstraction_prompt(q) for q in questions]
        abstract_outputs = self.batch_generate_vllm(abstract_prompts, self.params_abstract)
        abstract_queries = [out.outputs[0].text.strip() for out in abstract_outputs]

        # 2. Retrieve
        print("   🔍 Step 2: Retrieving rules...")
        query_embeddings = self.embedder.encode(abstract_queries, batch_size=64, show_progress_bar=True, convert_to_numpy=True).tolist()
        retrieved_batch = self.memory.batch_retrieve(query_embeddings, top_k=self.TOP_K, threshold=0.0)

        # 3. Generate RAG
        rag_prompts = []
        for i, q in enumerate(questions):
            rules_list = retrieved_batch[i]
            if rules_list:
                context_text = "\n".join([f"[Rule {k+1}]: {r[0]}" for k, r in enumerate(rules_list)])
                rag_prompts.append(self.construct_prompt(q, context_text))
            else:
                rag_prompts.append(self.construct_prompt(q, context=None))

        print("   ✍️  Step 3: Generating RAG answers...")
        rag_outputs = self.batch_generate_vllm(rag_prompts, self.params_generate)

        # ======================================================================
        # Phase 3: Evaluation & Adaptive Selection
        # ======================================================================
        correct_base = 0
        correct_pure_rag = 0
        correct_adaptive = 0
        
        adaptive_log = []

        print("\n⚖️  Calculating Metrics...")
        for i in range(total):
            gt_str = ground_truths[i]
            
            # 1. 解析 Baseline (n=6)
            base_ans, base_count, base_list = self.get_majority_vote(base_outputs[i])
            base_is_correct = self.check_correct(base_ans, gt_str)
            if base_is_correct: correct_base += 1

            # 2. 解析 RAG (n=3)
            rag_ans, rag_count, rag_list = self.get_majority_vote(rag_outputs[i])
            rag_is_correct = self.check_correct(rag_ans, gt_str)
            if rag_is_correct: correct_pure_rag += 1

            # 3. Adaptive Logic (保守策略)
            # 我们需要从 Baseline 的 6 次结果中取前 3 次来模拟 "Baseline (n=3)" 用于对比
            # 但这里我们直接用 Baseline (n=6) 的结果作为基础，如果 RAG 能打败 n=6 的 Baseline，那才是真的强
            
            # 为了实现 Adaptive 逻辑，我们需要 Baseline 的 "不确定性"
            # 如果 Baseline (n=6) 的票数很分散 (e.g. < 4/6)，说明 Baseline 不自信
            
            final_ans = base_ans
            selection_source = "Baseline"

            # 策略：
            # 条件 A: RAG 非常自信 (3/3) 且 答案与 Baseline 不同 -> 相信 RAG (强修正)
            # 条件 B: Baseline 不自信 (<4/6) 且 RAG 相对自信 (>=2/3) -> 相信 RAG (填补空白)
            
            if rag_ans is not None and rag_ans != base_ans:
                if rag_count == 3:
                    final_ans = rag_ans
                    selection_source = "RAG (Strong)"
                elif base_count < 4 and rag_count >= 2:
                    final_ans = rag_ans
                    selection_source = "RAG (Fill Gap)"
            
            # 检查 Adaptive 结果
            adaptive_is_correct = self.check_correct(final_ans, gt_str)
            if adaptive_is_correct: correct_adaptive += 1

            # 打印详细日志
            print(f"[{i+1}/{total}] Q: {questions[i][:50]}...")
            print(f"  Base(n=6): {base_ans} (Votes: {base_count}/{len(base_list)}) [{'✅' if base_is_correct else '❌'}]")
            print(f"  RAG (n=3): {rag_ans} (Votes: {rag_count}/{len(rag_list)}) [{'✅' if rag_is_correct else '❌'}]")
            print(f"  Strategy : {selection_source} -> Final: {final_ans} [{'✅' if adaptive_is_correct else '❌'}]")
            print("-" * 40)

            # 记录有趣的 Case (Baseline 错 -> RAG 对)
            if not base_is_correct and adaptive_is_correct:
                adaptive_log.append({
                    "q": questions[i],
                    "base": base_list,
                    "rag": rag_list,
                    "final": final_ans,
                    "gt": gt_str
                })

        # ======================================================================
        # Final Report
        # ======================================================================
        acc_base = correct_base / total * 100
        acc_pure_rag = correct_pure_rag / total * 100
        acc_adaptive = correct_adaptive / total * 100

        print("\n" + "="*60)
        print("🧪 Evaluation Results (Conservative/Adaptive Strategy)")
        print("="*60)
        print(f"1. Baseline (Majority Vote n=6): {acc_base:.2f}%")
        print(f"2. Pure RAG (Majority Vote n=3): {acc_pure_rag:.2f}%")
        print(f"3. Adaptive (Hybrid)           : {acc_adaptive:.2f}%  <-- Recommended")
        print("-" * 60)
        print(f"📈 Improvement over Baseline: {acc_adaptive - acc_base:+.2f}%")
        print("="*60)
        
        # 打印几个修正成功的例子
        if adaptive_log:
            print("\n🌟 Examples where Adaptive RAG fixed Baseline:")
            for item in adaptive_log[:3]:
                print(f"Q: {item['q'][:100]}...")
                print(f"   Base Votes: {item['base']} -> Wrong")
                print(f"   RAG Votes : {item['rag']} -> Correct")
                print("-" * 30)

def load_config(path="configurations/evaluate.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="configurations/evaluate.yaml", help="YAML config path")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluator = ScientificComparator(config)
    evaluator.run_scientific_test()
