from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载模型和 tokenizer
model_name = "Qwen/Qwen3-8B"  # 修正模型名称
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 使用量化解决显存问题
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,
    load_in_8bit=True  # 8bit量化，减少显存使用
)

# 加载 GSM8K 数据集
gsm8k = load_dataset("gsm8k", "main", split="test")

def generate_answer(question, model, tokenizer):
    # 获取模型所在的设备
    model_device = next(model.parameters()).device
    
    inputs = tokenizer(question, return_tensors="pt").to(model_device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,  # 设置pad token
            do_sample=False  # 使用贪婪解码加快速度
        )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

correct = 0
total = 5  # 测试前5个问题
print("\n Evaluation begins here. \n")

for i, sample in enumerate(gsm8k):
    if i >= total:
        break
        
    question = sample["question"]
    reference_answer = sample["answer"]
    
    print(f"\n--- 问题 {i+1} ---")
    print(f"问题: {question}")
    
    try:
        generated_answer = generate_answer(question, model, tokenizer)
        print(f"模型回答: {generated_answer}")
        print(f"参考答案: {reference_answer}")
        
        # 简单的答案匹配（可以根据需要改进）
        if reference_answer.strip() in generated_answer.strip():
            correct += 1
            print("✅ 回答正确")
        else:
            print("❌ 回答错误")
            
    except Exception as e:
        print(f"❌ 生成答案时出错: {e}")
        generated_answer = ""

accuracy = correct / total
print(f"\n最终准确率: {accuracy * 100:.2f}% ({correct}/{total})")