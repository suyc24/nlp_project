# shared_utils.py
import chromadb
from sentence_transformers import SentenceTransformer
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 数据库存储路径
DB_PATH = "./math_notebook_db"
COLLECTION_NAME = "math_notebook"

# 初始化 Embedding 模型 (只加载一次)
print("正在加载 Embedding 模型...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text):
    return embedder.encode(text).tolist()

def get_db_collection(reset=False):
    """
    获取向量数据库集合
    :param reset: 是否重置数据库 (仅在学习阶段使用 True)
    """
    # 使用 PersistentClient 实现数据持久化
    client = chromadb.PersistentClient(path=DB_PATH)
    
    if reset:
        try:
            client.delete_collection(name=COLLECTION_NAME)
            print(f"已重置集合: {COLLECTION_NAME}")
        except:
            pass
        collection = client.create_collection(name=COLLECTION_NAME)
    else:
        # 获取已存在的集合
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
    
    return collection