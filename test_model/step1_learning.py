# step1_learning.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import re
from openai import OpenAI
from datasets import load_dataset
from shared_utils import get_embedding, get_db_collection

client = OpenAI(
    api_key="sk-0549745c126c4db699b561da66c8c5df",
    base_url="https://api.deepseek.com"
)

def extract_kernel_with_deepseek(question, answer):
    """
    Teacher: DeepSeek 提取数学内核与普适规律
    返回: (abstract_key, python_skill)
    """
    system_prompt = """
You are a Professor of Mathematics and Computer Science. 
Your goal is to extract the **Universal Mathematical Laws** and **Problem Solving Kernels** from specific examples.

### Instructions:
1. **Ignore Surface Details**: Do not focus on "apples", "oranges", "John", or "money". Focus on the mathematical structure (e.g., "Set Theory", "Ratio", "Geometry: Midpoint Theorem", "Linear Equation").
2. **Abstract the Logic**: 
   - If you see a geometry problem with a midpoint, suggest "Doubling the Median" or "Midpoint Theorem".
   - If you see a problem about removing items, suggest "Set Subtraction: Total - Sum(Subsets)".
   - If you see a problem about rates, suggest "Work = Rate * Time".
3. **Generate Two Outputs**:
   - **Abstract Key**: A short, high-level description of the problem type (used for searching). NO specific numbers or nouns.
   - **Generic Python Skill**: A Python function that implements this universal logic.

### Output Format (JSON):
You must output a valid JSON object with exactly these two keys:
{
    "abstract_key": "The high-level mathematical description (e.g., 'Calculating remaining quantity after subtracting multiple subsets from a total')",
    "python_skill": "def solve(...): ... # The generic python code with docstrings explaining the law"
}
"""

    user_prompt = f"""
--- Specific Problem ---
Question: {question}
Solution: {answer}

--- Task ---
Extract the Universal Kernel and output the JSON.
"""
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat", # DeepSeek-V3
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1, # 低温以保证 JSON 格式稳定
            response_format={ "type": "json_object" }, # 强制 JSON 模式
            stream=False
        )
        content = response.choices[0].message.content.strip()
        
        # 解析 JSON
        data = json.loads(content)
        return data.get("abstract_key"), data.get("python_skill")
        
    except Exception as e:
        print(f"DeepSeek API/Parsing Error: {e}")
        # 简单的容错处理，如果 JSON 解析失败，尝试用正则提取
        return None, None

def main():
    # 1. 初始化数据库 (reset=True 表示清空旧数据重新学习)
    collection = get_db_collection(reset=True)
    
    # 2. 加载数据
    print("正在加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main")
    # 建议稍微多跑一点数据，因为抽象规律需要覆盖面
    train_subset = dataset['train'].shuffle(seed=42).select(range(50)) 

    print("\n=== 阶段一：深度学习 (Teacher: DeepSeek -> Abstract Kernel) ===")
    print("目标：提取普适规律，避免语义陷阱。\n")
    
    success_count = 0
    
    for i, item in enumerate(train_subset):
        question = item['question']
        answer = item['answer']
        
        print(f"[{i+1}/{len(train_subset)}] Analyzing: {question[:40]}...")
        
        # 调用 DeepSeek 提取内核
        abstract_key, skill_code = extract_kernel_with_deepseek(question, answer)
        
        if abstract_key and skill_code:
            # 清洗代码中的 Markdown 标记
            skill_code = skill_code.replace("```python", "").replace("```", "").strip()
            collection.add(
                documents=[skill_code], # 存储的内容：代码技能
                embeddings=[get_embedding(abstract_key)], # 索引的键：抽象数学描述
                metadatas=[{
                    "original_q": question, 
                    "abstract_desc": abstract_key, # 存下来方便调试看
                    "source": "deepseek_kernel"
                }],
                ids=[f"skill_{i}"]
            )
            print(f"   -> [Kernel Extracted]: {abstract_key}")
            success_count += 1
        else:
            print("   -> Failed to extract kernel.")

    print(f"\n笔记本构建完成！共提炼 {success_count} 条普适数学规律。")
    print("数据已保存至 ./math_notebook_db 文件夹。")

if __name__ == "__main__":
    main()