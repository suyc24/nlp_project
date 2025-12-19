import re
import torch
from collections import Counter
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
from vllm import LLM, SamplingParams
import time
import argparse 
import yaml



# ================= 1. Memory Manager (Unchanged) =================
class MemoryManager:
    # Initialize the MemoryManager.
    # This constructor sets up the ChromaDB client and retrieves the collection.
    def __init__(self, config):
        self.config = config
        self.MODEL_PATH = config["MODEL_PATH"]
        self.DB_PATH = config["DB_PATH"]
        self.GPU_UTILIZATION = config["GPU_UTILIZATION"]
        self.TOP_K = config["TOP_K"]
        self.SC_PATHS = config["SC_PATHS"]  
        self.RAG_THRESHOLD = config["RAG_THRESHOLD"]
        self.client = chromadb.PersistentClient(path=self.DB_PATH)
        self.collection = self.client.get_collection(name="rule_book") 
        
    # Retrieve the most similar documents for a batch of query embeddings.
    # Parameters:
    # - query_embeddings (list): A list of embeddings representing the queries.
    # - top_k (int): The number of top results to return for each query.
    # Purpose:
    # - Finds the most relevant stored rules or patterns for the given queries.
    # Returns:
    # - list: A list of lists, where each sublist contains tuples (doc, dist) for the top results.
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

# ================= 2. Scientific Comparator =================
class ScientificComparator:
    # Initialize the ScientificComparator.
    # This constructor sets up the vLLM engine, sampling parameters, and embedder.
    def __init__(self, config):
        print(f"🚀 Initializing vLLM engine (Rigorous Mode)...")
        self.config = config
        self.MODEL_PATH = config["MODEL_PATH"]
        self.DB_PATH = config["DB_PATH"]
        self.GPU_UTILIZATION = config["GPU_UTILIZATION"]
        self.TOP_K = config["TOP_K"]
        self.SC_PATHS = config["SC_PATHS"]  
        self.RAG_THRESHOLD = config["RAG_THRESHOLD"]
        self.llm = LLM(
            model=self.MODEL_PATH, 
            trust_remote_code=True,
            gpu_memory_utilization=self.GPU_UTILIZATION,
            tensor_parallel_size=1, 
            max_model_len=2048
        )
        
        # Define unified sampling strategy (SC)
        # Both Base and RAG models are given 5 chances for voting.
        self.params_sc = SamplingParams(
            n=self.SC_PATHS, 
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=256,
            stop=["<|endoftext|>", "<|im_end|>", "Question:"]
        )

        print("📥 Loading Embedder (CPU)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")
        self.memory = MemoryManager(config)

    # Construct a base prompt for the model.
    # Parameters:
    # - question (str): The question to be answered.
    # Purpose:
    # - Creates a simple chain-of-thought prompt for the model to generate an answer.
    # Returns:
    # - str: The constructed prompt.
    def construct_base_prompt(self, question):
        return f"<|im_start|>user\nQuestion: {question}\nLet's think step by step.\nAnswer:<|im_end|>\n<|im_start|>assistant\n"

    # Construct a RAG prompt for the model.
    # Parameters:
    # - question (str): The question to be answered.
    # - retrieved_items (list): A list of retrieved rules and their distances.
    # Purpose:
    # - Creates a prompt that includes relevant rules to guide the model's answer.
    # Returns:
    # - str: The constructed prompt.
    def construct_rag_prompt(self, question, retrieved_items):
        valid_items = [item[0] for item in retrieved_items if item[1] < self.RAG_THRESHOLD]
        if not valid_items:
            return self.construct_base_prompt(question)
        
        context_str = "\n".join([f"Rule {i+1}: {rule}" for i, rule in enumerate(valid_items)])
        prompt = f"""<|im_start|>user
You are a math expert. Here are some verified rules that might help solve the problem:
{context_str}

Question: {question}
Instruction: Solve the problem step-by-step. If any of the rules above apply, follow them strictly.
Answer:<|im_end|>
<|im_start|>assistant
"""
        return prompt

    # Extract the last number from a text string.
    # Parameters:
    # - text (str): The text to extract the number from.
    # Purpose:
    # - Parses the text to find and return the last numeric value.
    # Returns:
    # - float or None: The extracted number, or None if no number is found.
    def extract_answer(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    # Perform majority voting on the model's outputs.
    # Parameters:
    # - request_output (list): A list of outputs generated by the model.
    # Purpose:
    # - Aggregates the outputs to find the most common numeric answer.
    # Returns:
    # - float or None: The most common numeric answer, or None if no valid numbers are found.
    def majority_vote(self, request_output):
        valid_nums = []
        for output in request_output.outputs:
            num = self.extract_answer(output.text)
            if num is not None:
                valid_nums.append(num)
        if not valid_nums: return None
        return Counter(valid_nums).most_common(1)[0][0]

    # Check if the predicted answer matches the ground truth.
    # Parameters:
    # - pred (float): The predicted answer.
    # - gt_str (str): The ground truth answer as a string.
    # Purpose:
    # - Compares the predicted answer to the ground truth within a small tolerance.
    # Returns:
    # - bool: True if the prediction is correct, False otherwise.
    def check_correct(self, pred, gt_str):
        if "####" in gt_str:
            gold = self.extract_answer(gt_str.split("####")[1])
        else:
            gold = self.extract_answer(gt_str)
        if pred is None or gold is None: return False
        return abs(pred - gold) < 1e-4

    # Run the scientific test to evaluate the model's performance.
    # Purpose:
    # - Compares the accuracy of the Base and RAG models using Self-Consistency.
    # - Outputs the results and the contribution of the RAG model.
    def run_scientific_test(self):
        dataset = load_dataset("gsm8k", "main")["test"]
        questions = dataset["question"]
        ground_truths = dataset["answer"]
        total = len(questions)
        
        print(f"📊 Test set size: {total} | Sampling paths n={self.SC_PATHS} | Control variable: RAG Context")

        # ================= Phase 1: Base Model (Self-Consistency) =================
        print(f"\n🔵 [Group A] Base Model (Self-Consistency)...")
        base_prompts = [self.construct_base_prompt(q) for q in questions]
        
        t0 = time.time()
        base_outputs = self.llm.generate(base_prompts, self.params_sc, use_tqdm=True)
        print(f"   Time taken: {time.time()-t0:.2f}s")

        correct_base = 0
        for i, out in enumerate(base_outputs):
            pred = self.majority_vote(out)
            if self.check_correct(pred, ground_truths[i]):
                correct_base += 1
        
        acc_base = correct_base / total * 100
        print(f"   ✅ Base (SC) Accuracy: {acc_base:.2f}%")

        # ================= Phase 2: RAG Model (Self-Consistency) =================
        print(f"\n🟢 [Group B] RAG Model (Self-Consistency)...")
        
        # Pre-retrieval
        print("   -> Retrieving context...")
        q_embeddings = self.embedder.encode(questions, batch_size=64, show_progress_bar=True, convert_to_numpy=True).tolist()
        all_retrieved = self.memory.batch_retrieve(q_embeddings, top_k=self.TOP_K)
        
        rag_prompts = []
        for i, q in enumerate(questions):
            rag_prompts.append(self.construct_rag_prompt(q, all_retrieved[i]))

        t0 = time.time()
        rag_outputs = self.llm.generate(rag_prompts, self.params_sc, use_tqdm=True)
        print(f"   Time taken: {time.time()-t0:.2f}s")

        correct_rag = 0
        for i, out in enumerate(rag_outputs):
            pred = self.majority_vote(out)
            if self.check_correct(pred, ground_truths[i]):
                correct_rag += 1
        
        acc_rag = correct_rag / total * 100
        print(f"   ✅ RAG (SC) Accuracy: {acc_rag:.2f}%")

        # ================= Final Analysis =================
        print("\n" + "="*60)
        print("🧪 Scientific Attribution Analysis (Ablation Study)")
        print("="*60)
        print(f"Control variable: Self-Consistency (n={self.SC_PATHS})")
        print("-" * 60)
        print(f"1. Baseline Capability (Base + SC): {acc_base:.2f}%")
        print(f"2. Evolution Capability (Base + SC + RAG): {acc_rag:.2f}%")
        print("-" * 60)
        diff = acc_rag - acc_base
        print(f"📈 Pure RAG Gain: {diff:+.2f}%")
        
        if diff > 0:
            print("Conclusion: RAG provides effective information gain, not just randomness.")
        else:
            print("Conclusion: RAG failed to provide positive gain, possibly due to retrieval noise or ineffective prompt utilization.")
        print("="*60)
        

def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="config.yaml", help="YAML配置文件路径")
    args = parser.parse_args()
    config = load_config(args.config)
    evaluator = ScientificComparator(config)
    evaluator.run_scientific_test()