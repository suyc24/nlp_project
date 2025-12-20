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

# ================= 参数配置 =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DB_PATH = "./reflexion_full_db"

MAX_NEW_TOKENS = 1024
GPU_MEMORY_UTILIZATION = 0.90 

SC_PATHS = 1     
TOP_K = 3        
RAG_THRESHOLD = 0.35 


# 日志路径
RAG_LOG_PATH = "rag_usage_log.jsonl"
DEBUG_LOG_PATH = "debug_trace.jsonl"



# ====== IRCoT ====== 
IRCOT_MAX_STEPS = 6
IRCOT_K_PER_STEP = 3