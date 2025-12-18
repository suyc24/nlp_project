import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set HF Mirror to ensure stable downloads in certain regions
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Set proxy for HTTP and HTTPS requests
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

def extract_number(text):
    """
    Extracts the last numerical value from a given text string.
    
    This function is specifically designed for parsing answers from the GSM8K dataset,
    where the final answer is often the last number in the reasoning chain.
    
    Args:
        text (str): The input text containing numbers.
        
    Returns:
        float: The last number found in the text, or None if no number is found.
    """
    if not text: return None
    # Remove commas to handle numbers like "1,000" correctly
    text = text.replace(',', '')
    # Find all numbers (integers or decimals, positive or negative)
    matches = re.findall(r'-?\d+\.?\d*', text)
    # Return the last match converted to float
    return float(matches[-1]) if matches else None

def load_model_and_tokenizer(model_name):
    """
    Loads the pre-trained model and tokenizer from Hugging Face.
    
    Args:
        model_name (str): The path or name of the model in the Hugging Face Hub.
        
    Returns:
        tuple: A tuple containing (model, tokenizer).
            - model (AutoModelForCausalLM): The loaded causal language model.
            - tokenizer (AutoTokenizer): The loaded tokenizer.
    """
    print(f"Loading {model_name}...")
    # Load tokenizer with left padding (better for generation tasks)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    
    # Ensure pad_token is set (often missing in some models like Llama/Qwen)
    if not tokenizer.pad_token: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with automatic device mapping (GPU/CPU) and data type selection
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        dtype="auto", 
        device_map="auto"
    )
    return model, tokenizer

def construct_prompt(question):
    """
    Constructs a structured chat prompt for the GSM8K math reasoning task.
    
    This creates a list of messages compatible with the `apply_chat_template` method.
    
    Args:
        question (str): The math question to be asked.
        
    Returns:
        list: A list of dictionaries representing the conversation history.
              Example: [{"role": "system", ...}, {"role": "user", ...}]
    """
    return [
        {"role": "system", "content": "You are a helpful math assistant."},
        {"role": "user", "content": f"Question: {question}\nLet's think step by step and output the final answer."}
    ]

def batch_generate(model, tokenizer, prompts, max_new_tokens=512, temperature=0.2):
    """
    Generates responses for a batch of prompts using the provided model.
    
    Args:
        model (AutoModelForCausalLM): The language model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        prompts (list[str]): A list of formatted prompt strings (after applying chat template).
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
        temperature (float, optional): Sampling temperature. Lower values make output more deterministic. Defaults to 0.2.
        
    Returns:
        list[str]: A list of generated text strings (responses) corresponding to the input prompts.
    """
    # Tokenize inputs and move to the same device as the model
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Generate outputs without gradient calculation for efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            do_sample=True # Enable sampling to use temperature
        )
        
    # Decode the generated tokens, skipping the input tokens to get only the new response
    generated = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated

def generate_response(model, tokenizer, prompt, max_new_tokens=512, temperature=0.2):
    """
    Generates a single response for a given prompt string.
    
    Args:
        model (AutoModelForCausalLM): The language model to use for generation.
        tokenizer (AutoTokenizer): The tokenizer corresponding to the model.
        prompt (str): The input prompt string.
        max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 512.
        temperature (float, optional): Sampling temperature. Lower values make output more deterministic. Defaults to 0.2.
        
    Returns:
        str: The generated text string (response).
    """
    # Tokenize input and move to the same device as the model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(model.device)
    
    # Generate output without gradient calculation for efficiency
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature, 
            do_sample=True # Enable sampling to use temperature
        )
        
    # Decode the generated tokens, skipping the input tokens to get only the new response
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return generated
