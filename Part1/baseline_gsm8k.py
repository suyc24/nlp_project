import json
from datasets import load_dataset
from tqdm import tqdm
import utils

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
NUM_SAMPLES = 100
BATCH_SIZE = 16
OUTPUT_FILE = "baseline_gsm8k_results.json"

def main():
    model, tokenizer = utils.load_model_and_tokenizer(MODEL_NAME)

    print("Loading GSM8K...")
    dataset = load_dataset("gsm8k", "main", split=f"test[:{NUM_SAMPLES}]")
    
    results = []
    correct_count = 0
    
    print(f"Starting evaluation (Samples: {len(dataset)})...")
    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
        
        # Construct prompts using the utility function
        chat_prompts = [utils.construct_prompt(q) for q in batch['question']]
        
        # Apply chat template
        formatted_prompts = [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in chat_prompts
        ]
        
        # Generate responses
        generated = utils.batch_generate(model, tokenizer, formatted_prompts)
        
        for q, ans, gen in zip(batch['question'], batch['answer'], generated):
            gold = utils.extract_number(ans.split("####")[1] if "####" in ans else ans)
            pred = utils.extract_number(gen)
            is_correct = (gold is not None and pred is not None and abs(pred - gold) < 1e-4)
            
            if is_correct: correct_count += 1
            results.append({"question": q, "ground_truth": ans, "model_answer": gen, "correct": is_correct})

    acc = (correct_count / len(dataset)) * 100
    print(f"\nAccuracy: {acc:.2f}%")
    
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
