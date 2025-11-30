# step2_testing.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shared_utils import get_embedding, get_db_collection

# --- 配置 ---

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DISTANCE_THRESHOLD = 1 

# --- 加载模型 (Student) ---
print(f"正在加载本地模型: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype="auto", device_map="auto")

def call_qwen(prompt, max_tokens=512):
    """
    通用 Qwen 调用函数
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_len = model_inputs.input_ids.shape[1]
    
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            attention_mask=model_inputs.attention_mask,
            max_new_tokens=max_tokens,
            temperature=0.6, # 稍微降低温度，让逻辑更稳
            do_sample=True
        )
    
    generated_ids = [output_ids[input_len:] for output_ids in generated_ids]
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def generate_abstract_query(question):
    """
    查询重写：让 Student 模型把具体问题转化为抽象描述
    目的：为了更好地匹配 Step 1 中存储的 'Abstract Key'
    """
    prompt = f"""
Identify the core mathematical logic of the following problem. 
Ignore specific numbers and names. 
Output a short, single sentence description (e.g., "Calculating ratio between two groups", "Set subtraction with overlap").

Problem: {question}

Abstract Description:
"""
    # 限制输出长度，防止模型废话
    return call_qwen(prompt, max_tokens=50)

# --- RAG Prompt ---
RAG_PROMPT = """
I have retrieved a "Universal Mathematical Law" (Python Code) from your notebook that might be relevant.

--- Retrieved Law (Abstract Logic) ---
{retrieved_skill}
---

New Problem: {question}

### Instructions:
1. **Evaluate Relevance**: First, check if the Retrieved Law actually applies to the New Problem's logic.
   - The law might be about "Ratios" while the problem is about "Subtraction". If so, IGNORE the law.
2. **Apply or Discard**:
   - If relevant: Use the logic in the Python code to guide your step-by-step solution.
   - If NOT relevant: Solve the problem entirely on your own logic.
3. **Solve**: Provide the step-by-step solution and the final answer.
"""

def main():
    # 1. 连接数据库
    collection = get_db_collection(reset=False)
    count = collection.count()
    if count == 0:
        print("错误：数据库为空！请先运行 step1_learning.py")
        return
    print(f"成功连接笔记本，当前拥有 {count} 条普适规律。")

    # 2. 加载测试数据
    dataset = load_dataset("gsm8k", "main")
    test_subset = dataset['test'].shuffle(seed=42).select(range(20))

    print("\n=== 阶段二：测试 (Student: Qwen + Abstract Notebook) ===")
    results_data = []

    for i, item in enumerate(test_subset):
        question = item['question']
        ground_truth = item['answer']
        
        print(f"\nTest Q{i+1}: {question[:50]}...")
        
        # --- 1. Baseline (裸考) ---
        start_time = time.time()
        base_ans = call_qwen(f"Solve this math problem step-by-step: {question}")
        print(f"   -> Baseline: {time.time() - start_time:.2f}s")
        
        # --- 2. Retrieval (检索) ---
        # A. 查询重写：具体 -> 抽象
        abstract_query = generate_abstract_query(question)
        print(f"   -> [Query Rewrite]: {abstract_query}")
        
        # B. 使用抽象描述进行检索
        query_embed = get_embedding(abstract_query)
        results = collection.query(query_embeddings=[query_embed], n_results=1)
        
        retrieved_skill = "# No relevant skill found."
        retrieved_abstract_desc = "N/A"
        
        # C. 检查结果
        if results['documents'] and results['documents'][0]:
            distance = results['distances'][0][0]
            # 获取 Step 1 存进去的抽象描述，方便人工检查匹配是否合理
            retrieved_abstract_desc = results['metadatas'][0][0].get('abstract_desc', 'N/A')
            
            if distance < DISTANCE_THRESHOLD:
                retrieved_skill = results['documents'][0][0]
                print(f"   -> [Match Found] (Dist: {distance:.4f})")
                print(f"      Query Logic: {abstract_query}")
                print(f"      Found Logic: {retrieved_abstract_desc}")
            else:
                print(f"   -> [Discarded] Too far (Dist: {distance:.4f} > {DISTANCE_THRESHOLD})")
                print(f"      Found Logic: {retrieved_abstract_desc}")
        
        # --- 3. RAG Inference (推理) ---
        start_time = time.time()
        rag_ans = call_qwen(RAG_PROMPT.format(retrieved_skill=retrieved_skill, question=question))
        print(f"   -> Notebook: {time.time() - start_time:.2f}s")

        results_data.append({
            "question": question,
            "ground_truth": ground_truth,
            "baseline_answer": base_ans,
            "notebook_answer": rag_ans,
            "abstract_query_generated": abstract_query, # 记录生成的抽象查询
            "retrieved_abstract_desc": retrieved_abstract_desc, # 记录检索到的抽象描述
            "retrieved_skill": retrieved_skill
        })

    # 保存结果
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print("\n结果已保存到 experiment_results.json")

if __name__ == "__main__":
    main()