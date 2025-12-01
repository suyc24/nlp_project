# step1_learning.py
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import load_dataset
from tqdm import tqdm # 进度条库
from shared_utils import get_embedding, get_db_collection

# --- 配置 ---

client = OpenAI(
    api_key="sk-0549745c126c4db699b561da66c8c5df", 
    base_url="https://api.deepseek.com"
)

# 并发线程数 (DeepSeek API 限速较高，可以开 10-20 线程，根据你的账号等级调整)
MAX_WORKERS = 10 
# 多少条数据存一次盘 (ChromaDB 本地版不需要显式 commit，但我们可以控制打印频率)
LOG_INTERVAL = 10

# 用于去重的全局集合 (线程安全需要注意，但在 Python GIL 下简单的 set add 是原子的，或者用 Lock)
seen_abstract_keys = set()
lock = threading.Lock()

def extract_kernel_with_deepseek(question, answer):
    """
    Teacher: DeepSeek 提取数学内核
    """
    system_prompt = """
You are a Professor of Mathematics. Extract the **Universal Mathematical Kernel** from the problem.
Ignore surface details (names, items). Focus on the logic (Set Theory, Ratio, Geometry, etc.).

Output JSON with two keys:
1. "abstract_key": A short, high-level description (e.g., "Calculating remaining quantity after subtracting subsets").
2. "python_skill": A generic Python function implementing this logic.
"""
    user_prompt = f"Question: {question}\nSolution: {answer}\n\nExtract Kernel (JSON):"
    
    try:
        # 简单的重试机制
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    response_format={ "type": "json_object" },
                    timeout=30 # 设置超时
                )
                content = response.choices[0].message.content.strip()
                data = json.loads(content)
                return data.get("abstract_key"), data.get("python_skill")
            except Exception as e:
                time.sleep(1) # 失败等待 1 秒
                continue
        return None, None
    except Exception:
        return None, None

def process_item(item, index, collection):
    """
    处理单条数据的 Worker 函数
    """
    question = item['question']
    answer = item['answer']
    doc_id = f"skill_{index}" # 使用原始数据集的索引作为 ID，方便断点续传

    # 1. 提取内核
    abstract_key, skill_code = extract_kernel_with_deepseek(question, answer)
    
    if not abstract_key or not skill_code:
        return False, "API Error"

    # 2. 简单的去重逻辑
    # 如果这个抽象规律已经出现过（完全一样的字符串），则跳过
    # 注意：这里用 Lock 保证线程安全
    with lock:
        if abstract_key in seen_abstract_keys:
            return False, "Duplicate"
        seen_abstract_keys.add(abstract_key)

    # 3. 清洗代码
    skill_code = skill_code.replace("```python", "").replace("```", "").strip()

    # 4. 写入数据库 (ChromaDB 的 add 不是完全线程安全的，建议加锁或在主线程批量写入)
    # 为了简化代码，这里加锁写入。虽然会降低一点点速度，但保证数据安全。
    with lock:
        collection.add(
            documents=[skill_code],
            embeddings=[get_embedding(abstract_key)], 
            metadatas=[{
                "original_q": question, 
                "abstract_desc": abstract_key,
                "source": "deepseek_kernel"
            }],
            ids=[doc_id]
        )
    
    return True, abstract_key

def main():
    # 1. 连接数据库 (reset=False! 这样才能断点续传)
    # 如果你想彻底重跑，请手动删除 ./math_notebook_db 文件夹
    collection = get_db_collection(reset=False)
    
    # 2. 加载全量数据
    print("正在加载 GSM8K 全量数据集...")
    dataset = load_dataset("gsm8k", "main")
    train_data = dataset['train'] # 不再 shuffle 和 select，使用全量
    total_items = len(train_data)
    
    # 3. 断点续传检查
    print("检查已处理的数据...")
    existing_ids = set(collection.get()['ids'])
    print(f"数据库中已有 {len(existing_ids)} 条数据。")
    
    # 预加载已有的 abstract_key 防止重启后重复添加相同的规律
    if len(existing_ids) > 0:
        existing_data = collection.get(include=['metadatas'])
        for meta in existing_data['metadatas']:
            if meta and 'abstract_desc' in meta:
                seen_abstract_keys.add(meta['abstract_desc'])
        print(f"已加载 {len(seen_abstract_keys)} 条去重规则。")

    # 找出还需要处理的索引
    indices_to_process = [i for i in range(total_items) if f"skill_{i}" not in existing_ids]
    print(f"剩余 {len(indices_to_process)} 条数据待处理。")

    if not indices_to_process:
        print("所有数据已处理完毕！")
        return

    print(f"\n=== 开始全量学习 (Threads: {MAX_WORKERS}) ===")
    
    # 使用 ThreadPoolExecutor 进行并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交任务
        future_to_idx = {
            executor.submit(process_item, train_data[i], i, collection): i 
            for i in indices_to_process
        }
        
        # 使用 tqdm 显示进度条
        success_count = 0
        duplicate_count = 0
        error_count = 0
        
        pbar = tqdm(as_completed(future_to_idx), total=len(indices_to_process), unit="item")
        
        for future in pbar:
            try:
                success, msg = future.result()
                if success:
                    success_count += 1
                elif msg == "Duplicate":
                    duplicate_count += 1
                else:
                    error_count += 1
                
                # 更新进度条描述
                pbar.set_description(f"✅{success_count} ♻️{duplicate_count} ❌{error_count}")
                
            except Exception as e:
                error_count += 1
                print(f"\nCritical Error: {e}")

    print(f"\n学习结束！")
    print(f"新增有效规律: {success_count}")
    print(f"跳过重复规律: {duplicate_count}")
    print(f"API/解析失败: {error_count}")
    print(f"当前数据库总容量: {collection.count()}")

if __name__ == "__main__":
    main()