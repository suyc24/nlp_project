"""
Entry script to generate step-by-step GSM8K answers with Qwen models.
This script loads GSM8K from Hugging Face, builds CoT prompts, and runs generation.
Configuration is defined via global variables (no CLI args).
"""

from typing import List

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import extract_answer_from_response, check_answer_correctness


# ==================== Global Configuration ====================

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATA_SPLIT = "test"  # train or test
LIMIT = 20  # number of samples to run; set None or 0 for full split
MAX_TOKENS = 256
TEMPERATURE = 0.2
DEVICE = "cuda"


# Load GSM8K split with optional limit
# Args:
#   split: Dataset split name (train/test)
#   limit: Maximum number of samples to return
# Returns:
#   List of dataset rows
def load_gsm8k(split: str = "test", limit: int = 20):
    ds = load_dataset("gsm8k", "main", split=split)
    if limit and limit > 0:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


# Build CoT prompts with simple instruction
def build_inputs(questions: List[str]) -> List[str]:
    return [f"Let's solve this step by step:\n\n{q.strip()}" for q in questions]


# Generate chain-of-thought completions
def generate_cot(model, tokenizer, prompts: List[str], max_tokens: int, temperature: float):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            eos_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # Strip prompts from completions to keep only generated text
    completions = []
    for prompt, full_text in zip(prompts, decoded):
        completions.append(full_text[len(prompt):].strip())
    return completions


# Orchestrate loading, prompting, generation, and evaluation
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    data = load_gsm8k(split=DATA_SPLIT, limit=LIMIT)
    questions = [row["question"] for row in data]
    ground_truth = [extract_answer_from_response(row["answer"]) for row in data]

    prompts = build_inputs(questions)
    completions = generate_cot(model, tokenizer, prompts, MAX_TOKENS, TEMPERATURE)

    correct = 0
    for idx, (q, gt, comp) in enumerate(zip(questions, ground_truth, completions)):
        pred_ans = extract_answer_from_response(comp)
        is_correct = check_answer_correctness(pred_ans, gt)
        correct += int(is_correct)
        print(f"\n=== Sample {idx} ===")
        print(f"Question: {q}")
        print(f"Ground Truth: {gt}")
        print(f"Completion:\n{comp}")
        print(f"Predicted Answer: {pred_ans} | Correct: {is_correct}")

    total = len(completions)
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")


if __name__ == "__main__":
    main()
