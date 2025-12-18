# step2_testing.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
import json
import time
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from shared_utils import get_embedding, get_db_collection
from sentence_transformers import SentenceTransformer

# --- 配置 ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
DISTANCE_THRESHOLD = 1
BATCH_SIZE = 32 # 保持批量加速
SAVE_INTERVAL = 50 
OUTPUT_FILE = "gsm8k_optimized_results.json"

# --- 1. 加载模型 ---
print(f"正在加载本地模型: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, 
    device_map="auto"
)

print("正在加载 Embedding 模型...")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- 2. 批量推理函数 ---
def batch_generate(prompts, max_new_tokens=512, temperature=0.2):
    """
    通用批量生成
    """
    formatted_prompts = []
    for p in prompts:
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": p}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)

    inputs = tokenizer(
        formatted_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)

    input_len = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature, # 降温，提高稳定性
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )

    decoded_outputs = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
    return [out.strip() for out in decoded_outputs]

# --- 3. 评分工具 ---
def extract_number(text):
    if not text: return None
    text = text.replace(',', '')
    matches = re.findall(r'-?\d+\.?\d*', text)
    if matches: return float(matches[-1])
    return None

def is_correct(model_output, ground_truth_text):
    if "####" in ground_truth_text:
        gold_num = extract_number(ground_truth_text.split("####")[1])
    else:
        gold_num = extract_number(ground_truth_text)
    
    pred_num = extract_number(model_output)
    if pred_num is None or gold_num is None: return False
    return abs(pred_num - gold_num) < 1e-4

# --- 4. 主流程 ---
def main():
    collection = get_db_collection(reset=False)
    if collection.count() == 0:
        print("错误：数据库为空！")
        return

    print("正在加载 GSM8K 全量测试集...")
    dataset = load_dataset("gsm8k", "main")
    test_data = dataset['test']
    
    total_data = len(test_data)
    results_data = []
    
    baseline_correct = 0
    notebook_correct = 0

    print(f"\n=== 开始优化后的测试 (Batch Size: {BATCH_SIZE}) ===")
    
    pbar = tqdm(range(0, total_data, BATCH_SIZE), total=total_data // BATCH_SIZE + 1, unit="batch")

    for i in pbar:
        # 准备 Batch
        batch_slice = test_data.select(range(i, min(i + BATCH_SIZE, total_data)))
        questions = batch_slice['question']
        ground_truths = batch_slice['answer']
        
        # --- A. Baseline ---
        # 简单的 CoT Prompt
        base_prompts = [f"Question: {q}\nLet's think step by step and output the final answer." for q in questions]
        base_answers = batch_generate(base_prompts, temperature=0.2)
        
        # --- B. 优化后的 Query Rewrite (One-Shot) ---
        # 增加一个示例，教 0.5B 模型怎么做抽象
        # 目的：提高抽象质量，从而提高检索准确率
        rewrite_template = """
Task: Describe the mathematical logic of the problem in one short sentence. Ignore numbers and names.

Example:
Problem: John has 5 apples and eats 2. How many are left?
Logic: Subtracting a subset from a total quantity.

Problem: {q}
Logic:
"""
        rewrite_prompts = [rewrite_template.format(q=q) for q in questions]
        abstract_queries = batch_generate(rewrite_prompts, max_new_tokens=32, temperature=0.2)
        
        # --- C. 向量检索 ---
        query_embeddings = embedder.encode(abstract_queries).tolist()
        rag_prompts = []
        retrieved_infos = [] 
        
        for idx, q_embed in enumerate(query_embeddings):
            results = collection.query(query_embeddings=[q_embed], n_results=1)
            
            skill = ""
            desc = "N/A"
            dist = 999
            
            if results['documents'] and results['documents'][0]:
                d = results['distances'][0][0]
                if d < DISTANCE_THRESHOLD:
                    skill = results['documents'][0][0]
                    desc = results['metadatas'][0][0].get('abstract_desc', 'N/A')
                    dist = d
            
            retrieved_infos.append({"desc": desc, "dist": dist})
            
            # --- D. 优化后的 RAG Prompt (Variable Mapping) ---
            # 策略：不让模型去评价相关性，而是让它扮演“代码执行者”
            # 强迫它把题目数字代入模板，这比“推理”简单
            if skill:
                rag_p = f"""
Refence Python Logic:
```python
{skill}
```

Task:
1. Read the Reference Python Logic above.
2. Identify the variables in the New Problem below that match the logic.
3. Calculate the result step-by-step using the New Problem's numbers.

New Problem: {questions[idx]}
Solution:
"""
            else:
                # 没搜到就裸考
                rag_p = f"Question: {questions[idx]}\nLet's think step by step and output the final answer."
            
            rag_prompts.append(rag_p)
            
        # --- E. RAG 推理 ---
        rag_answers = batch_generate(rag_prompts, temperature=0.2)
        
        # --- F. 评分 ---
        for j in range(len(questions)):
            if is_correct(base_answers[j], ground_truths[j]):
                baseline_correct += 1
            
            if is_correct(rag_answers[j], ground_truths[j]):
                notebook_correct += 1
                
            results_data.append({
                "id": i + j,
                "question": questions[j],
                "ground_truth": ground_truths[j],
                "baseline": base_answers[j],
                "notebook": rag_answers[j],
                "abstract_query": abstract_queries[j],
                "retrieved_desc": retrieved_infos[j]['desc'],
                "distance": retrieved_infos[j]['dist']
            })

        # 进度更新
        processed = min(i + BATCH_SIZE, total_data)
        base_acc = (baseline_correct / processed) * 100
        note_acc = (notebook_correct / processed) * 100
        pbar.set_description(f"Base: {base_acc:.1f}% | Note: {note_acc:.1f}%")

        if len(results_data) % SAVE_INTERVAL == 0:
             with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)

    # 最终报告
    final_base_acc = (baseline_correct / total_data) * 100
    final_note_acc = (notebook_correct / total_data) * 100
    
    print("\n" + "="*40)
    print(f"Total: {total_data}")
    print(f"Base Acc: {final_base_acc:.2f}%")
    print(f"Note Acc: {final_note_acc:.2f}%")
    print(f"Lift: {final_note_acc - final_base_acc:+.2f}%")
    print("="*40)
    
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()