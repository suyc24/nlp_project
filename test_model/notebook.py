import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from openai import OpenAI
import json
import time
import chromadb
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

client = OpenAI(
    api_key="sk-0549745c126c4db699b561da66c8c5df", 
    base_url="https://api.deepseek.com"
)

# --- Qwen (Student) 配置 ---
model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"正在加载本地模型: {model_name} ...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")

# --- 向量数据库配置 ---
chroma_client = chromadb.Client()
# 每次运行前重置 collection，防止数据重复堆积
try:
    chroma_client.delete_collection(name="math_notebook")
except:
    pass
collection = chroma_client.create_collection(name="math_notebook")

print("正在加载 Embedding 模型...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

DISTANCE_THRESHOLD = 0 #这个参数值需要再实验

# ==========================================
# 2. 工具函数定义
# ==========================================

def get_embedding(text):
    return embedder.encode(text).tolist()

def call_qwen(prompt):
    """
    调用本地 Qwen 模型进行推理 (Student)
    """
    messages = [
        {"role": "system", "content": "You are a helpful math assistant. You are good at following code logic to solve problems."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    input_len = model_inputs.input_ids.shape[1]
    
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        max_new_tokens=512,
        temperature=0.7, # 稍微增加一点创造性，避免死板
        do_sample=True
    )
    
    generated_ids = [
        output_ids[input_len:] for output_ids in generated_ids
    ]
    
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

def extract_skill_with_deepseek(question, answer):
    """
    调用 DeepSeek API 提取通用 Python 技能 (Teacher)
    """
    system_prompt = """
You are an expert Math Teacher and Python Programmer.
Your goal is to abstract a specific math problem into a generic "Python Skill".

Rules:
1. Analyze the logic of the provided Problem and Solution.
2. Create a generic Python function that solves this TYPE of problem.
3. Replace specific numbers with variables (e.g., apple_count, price_per_unit).
4. The function should take inputs and return the final answer.
5. Add comments explaining the logic.
6. IMPORTANT: ONLY output the Python code block. Do not output any explanations or markdown text like 'Here is the code'.
"""

    user_prompt = f"""
Problem: {question}
Ground Truth Solution: {answer}

Generate the generic Python Skill:
"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # 使用 V3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # 保持逻辑严谨
            stream=False
        )
        content = response.choices[0].message.content.strip()
        
        # 清洗数据：移除 Markdown 标记
        content = content.replace("```python", "").replace("```", "").strip()
        return content
    except Exception as e:
        print(f"DeepSeek API Error: {e}")
        return None

# ==========================================
# 3. Prompt 模板
# ==========================================

# 推理 Prompt：让 Qwen 参考 Python 代码逻辑
RAG_PROMPT = """
I have retrieved a Python code template that solves a similar problem type from your notebook. 
Please read the code to understand the underlying logic, and then solve the New Problem.

--- Retrieved Logic (Python Template) ---
{retrieved_skill}
---

New Problem: {question}

Instructions:
1. Analyze the New Problem.
2. Map the values in the New Problem to the logic in the Python Template.
3. Solve the problem step-by-step.
4. Provide the final answer.
"""

# ==========================================
# 4. 主流程
# ==========================================

# 加载数据
print("正在加载 GSM8K 数据集...")
dataset = load_dataset("gsm8k", "main")
seed = 42
train_subset = dataset['train'].shuffle(seed=seed).select(range(50)) # 随机学习 50 题
test_subset = dataset['test'].shuffle(seed=seed).select(range(20))  # 随机测试 20 题

# --- 阶段一：学习 (Building Notebook with DeepSeek) ---
print("\n=== 阶段一：学习 (Teacher: DeepSeek -> Notebook) ===")

for i, item in enumerate(train_subset):
    question = item['question']
    answer = item['answer']
    
    print(f"[{i+1}/{len(train_subset)}] Learning from: {question[:40]}...")
    
    # 1. Teacher (DeepSeek) 提炼技能
    skill = extract_skill_with_deepseek(question, answer)
    
    if skill:
        # 2. 存入向量库
        collection.add(
            documents=[skill],
            embeddings=[get_embedding(question)], 
            metadatas=[{"original_q": question, "source": "deepseek_v3"}],
            ids=[f"skill_{i}"]
        )
        # 打印一下生成的代码头几行看看效果
        first_line = skill.split('\n')[0]
        print(f"   -> Generated Skill: {first_line} ... (Length: {len(skill)})")
    else:
        print("   -> Failed to generate skill.")
        

print(f"\n笔记本构建完成，共存储 {collection.count()} 条高质量 Python 经验。")

# --- 阶段二：测试 (Testing with Qwen) ---
print("\n=== 阶段二：测试 (Student: Qwen + Notebook) ===")

results_data = []

for i, item in enumerate(test_subset):
    question = item['question']
    ground_truth = item['answer']
    
    print(f"\nTest Q{i+1}: {question[:50]}...")
    
    # 1. Baseline (裸考)
    start_time = time.time()
    base_ans = call_qwen(f"Solve this math problem step-by-step: {question}")
    print(f"   -> Baseline generated ({time.time() - start_time:.2f}s)")
    
    # 2. Notebook (查笔记)
    query_embed = get_embedding(question)
    results = collection.query(query_embeddings=[query_embed], n_results=1) #这里以后n_results可以改
    
    if results['documents'] and results['documents'][0] and results['distances'][0][0] > DISTANCE_THRESHOLD:
        retrieved_skill = results['documents'][0][0]
        print(f"   -> Retrieved Skill Source: {results['metadatas'][0][0]['original_q'][:40]}...")
    else:
        retrieved_skill = "# No relevant skill found."
        print("   -> No skill retrieved.")
    
    start_time = time.time()
    rag_ans = call_qwen(RAG_PROMPT.format(retrieved_skill=retrieved_skill, question=question))
    print(f"   -> Notebook generated ({time.time() - start_time:.2f}s)")

    # 存入列表
    results_data.append({
        "question": question,
        "ground_truth": ground_truth,
        "baseline_answer": base_ans,
        "notebook_answer": rag_ans,
        "retrieved_skill": retrieved_skill
    })

# 保存结果
output_file = "experiment_results.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results_data, f, indent=2, ensure_ascii=False)

print(f"\n实验结束！结果已保存到 '{output_file}'")