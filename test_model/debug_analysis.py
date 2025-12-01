import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from shared_utils import get_embedding, get_db_collection
from sentence_transformers import SentenceTransformer

# --- 配置 ---
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DISTANCE_THRESHOLD = 1
DEBUG_SAMPLES = 10  # 只分析 10 条数据

# --- 加载模型 ---
print(f"正在加载模型: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, padding_side='left')
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# --- 工具函数 ---
def call_qwen(prompt, max_new_tokens=256):
    messages = [{"role": "system", "content": "You are a helpful math assistant."}, {"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.1, do_sample=False)
    return tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def main():
    collection = get_db_collection(reset=False)
    print(f"📚 知识库大小: {collection.count()}")

    dataset = load_dataset("gsm8k", "main")
    test_data = dataset['test'].shuffle(seed=42).select(range(50)) # 先取50条跑跑看

    print("\n🔍 开始深度调试分析 (寻找 Base 对但 Note 错的案例)...")
    
    analyzed_count = 0
    html_content = "<html><style>body{font-family:sans-serif; max-width:800px; margin:auto; padding:20px;} .case{border:1px solid #ccc; margin-bottom:20px; padding:15px; border-radius:8px;} .good{color:green;} .bad{color:red;} .code{background:#f4f4f4; padding:10px; font-family:monospace; white-space:pre-wrap;}</style><body><h1>Debugging Report</h1>"

    for i, item in enumerate(test_data):
        if analyzed_count >= DEBUG_SAMPLES: break

        question = item['question']
        ground_truth = item['answer']

        # 1. Baseline
        base_ans = call_qwen(f"Question: {question}\nLet's think step by step and output the final answer.")
        
        # 2. RAG Process
        # A. Query Rewrite
        rewrite_prompt = f"""Task: Describe the math logic. Ignore numbers.
Example: John has 5 apples, eats 2. -> Logic: Subtraction.
Problem: {question}
Logic:"""
        abstract_query = call_qwen(rewrite_prompt, max_new_tokens=32)

        # B. Retrieve
        query_embed = embedder.encode(abstract_query).tolist()
        results = collection.query(query_embeddings=[query_embed], n_results=1)
        
        retrieved_desc = "N/A"
        skill_code = ""
        dist = 999
        
        if results['documents'] and results['documents'][0]:
            dist = results['distances'][0][0]
            if dist < DISTANCE_THRESHOLD:
                skill_code = results['documents'][0][0]
                retrieved_desc = results['metadatas'][0][0].get('abstract_desc', 'N/A')

        # C. RAG Generation
        if skill_code:
            rag_prompt = f"""Reference Code:
```python
{skill_code}
```
Task: Solve the problem using the logic above.
Problem: {question}
Solution:"""
        else:
            rag_prompt = f"Question: {question}\nLet's think step by step."
            
        rag_ans = call_qwen(rag_prompt)

        # 简单的正确性判断 (看答案是否包含 Ground Truth 里的数字)
        # 这里仅用于快速筛选 "翻车" 案例
        gold_num = ground_truth.split("####")[-1].strip()
        base_correct = gold_num in base_ans
        note_correct = gold_num in rag_ans

        # 🎯 只关注：Base 对了，但 Note 错了的案列 (或者 Note 检索极其离谱的案例)
        if True: # 把这里改成 if base_correct and not note_correct: 可以只看翻车案例
            analyzed_count += 1
            print(f"\n[{analyzed_count}] Analyzing Q: {question[:30]}...")
            
            # 控制台简单打印
            print(f"   Abstract Query: {abstract_query}")
            print(f"   Retrieved Desc: {retrieved_desc} (Dist: {dist:.4f})")
            print(f"   Base Correct: {base_correct} | Note Correct: {note_correct}")

            # HTML 详细报告
            color = "#ffe6e6" if (base_correct and not note_correct) else "#f9f9f9"
            html_content += f"""
            <div class='case' style='background:{color}'>
                <h3>Question: {question}</h3>
                <p><b>Ground Truth:</b> {gold_num}</p>
                <hr>
                <p><b>1. Abstract Query (0.5B generated):</b><br>{abstract_query}</p>
                <p><b>2. Retrieval (Dist: {dist:.4f}):</b><br>
                   Key: {retrieved_desc}<br>
                   <div class='code'>{skill_code[:200]}...</div>
                </p>
                <hr>
                <p><b>3. Baseline Output:</b><br>{base_ans}</p>
                <p><b>4. Notebook Output:</b><br>{rag_ans}</p>
                <p><b>Result:</b> Base <span class='{ "good" if base_correct else "bad" }'>{base_correct}</span> vs Note <span class='{ "good" if note_correct else "bad" }'>{note_correct}</span></p>
            </div>
            """

    html_content += "</body></html>"
    with open("debug_report.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("\n✅ 调试完成！请打开 'debug_report.html' 查看详细分析。")

if __name__ == "__main__":
    main()