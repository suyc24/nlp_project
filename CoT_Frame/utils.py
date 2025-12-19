"""
Utility functions for multi-step reasoning with CoT (Chain-of-Thought).
This module provides only common tools for answer extraction, step decomposition,
and basic validation.
"""

import re
import torch
from typing import List

# ==================== Answer Extraction ====================

INVALID_ANS = "[invalid]"

# Normalize numeric string formatting
def standardize_value_str(x: str) -> str:
    y = x.replace(",", "")
    if '.' in y:
        y = y.rstrip('0')
        if y[-1] == '.':
            y = y[:-1]
    if not len(y):
        return INVALID_ANS
    if y[0] == '.':
        y = '0' + y
    if y[-1] == '%':
        y = str(eval(y[:-1]) / 100)
    return y.rstrip('.')

def extract_boxed_answers(text):
    answers = []
    for piece in text.split('boxed{')[1:]:
        n = 0
        for i in range(len(piece)):
            if piece[i] == '{':
                n += 1
            elif piece[i] == '}':
                n -= 1
                if n < 0:
                    if i + 1 < len(piece) and piece[i + 1] == '%':
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers

def extract_answer_robust(pred_str: str) -> str:
    """
    Robust answer extraction that handles \\boxed{}, "The answer is", and last number fallback.
    """
    pred = []
    if 'boxed' in pred_str:
        pred = extract_boxed_answers(pred_str)
    elif ('he answer is' in pred_str):
        pred = [pred_str.split('he answer is')[-1].strip()]
    else:
        # Fallback: look for the last number
        pattern = '-?\d*\.?\d+'
        ans = re.findall(pattern, pred_str.replace(",", ""))
        if(len(ans) >= 1):
            ans = ans[-1]
            if ans:
                pred.append(ans)

    if not pred:
        return INVALID_ANS
        
    # Clean up the extracted answer
    ans = pred[-1]
    ans = ans.strip().split("\n")[0]
    ans = ans.lstrip(":")
    ans = ans.rstrip(".")
    ans = ans.replace(",", "") # Standardize: remove commas
    return standardize_value_str(ans)

# Extract answer from GSM8K formatted response (Ground Truth)
def extract_answer_from_response(response: str) -> str:
    if "####" in response:
        answer = response.split("####")[-1].strip()
        return standardize_value_str(answer)
    return extract_answer_robust(response)

# Check predicted answer correctness
def check_answer_correctness(predicted: str, ground_truth: str) -> bool:
    if predicted == INVALID_ANS:
        return False
    return str(predicted) == str(ground_truth)


# ==================== Step Decomposition ====================

def split_reasoning_into_steps(reasoning: str) -> List[str]:
    steps = [step.strip() for step in reasoning.split("\n") if step.strip()]
    return steps

def merge_short_steps(steps: List[str], min_length: int = 20) -> List[str]:
    if not steps:
        return []
    
    merged = []
    current_step = steps[0]
    
    for i in range(1, len(steps)):
        if len(current_step) < min_length:
            current_step += "\n" + steps[i]
        else:
            merged.append(current_step)
            current_step = steps[i]
    
    if current_step:
        merged.append(current_step)
    
    return merged

# ==================== Self-Explore Verification ====================

def verify_step(model, tokenizer, prompt, ground_truth, k=5, max_new_tokens=128):
    """
    Verifies if a step is valid by generating k completions from the current prompt
    and checking if any of them lead to the correct answer.
    """
    # Replicate prompt k times
    inputs = tokenizer([prompt] * k, return_tensors="pt", padding=True).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7, # Higher temperature for exploration
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    # Check if any completion contains the correct answer
    for full_text in decoded:
        generated_part = full_text[len(prompt):]
        # Use robust extraction
        pred = extract_answer_robust(generated_part)
        
        # Check correctness
        if check_answer_correctness(pred, ground_truth):
            return True
            
    return False
