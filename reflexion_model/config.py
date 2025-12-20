import os

# ================= 环境变量配置 (必须最先执行) =================
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)

os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# vLLM 多进程设置
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ================= 训练参数配置 =================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct-AWQ"
DB_PATH = "./reflexion_full_db"
CHUNK_SIZE = 64          # 处理单元大小
MAX_NEW_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.90 
TARGET_ACCURACY = 75.0   # 目标准确率
MAX_EPOCHS = 5           # 最大训练轮数

# 日志路径
RAG_LOG_PATH = "rag_usage_log.jsonl"
DEBUG_LOG_PATH = "debug_trace.jsonl"