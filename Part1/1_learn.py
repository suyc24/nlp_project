import utils
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    # Load model and data
    model, tokenizer = utils.load_model_and_tokenizer(MODEL_NAME)
    data = load_dataset("gsm8k", "main", split="test")[0]

    # Custom prompt template (edit this to customize)
    custom_prompt_template = """You are a helpful math assistant. 
    Question: {question}
    
    Instruct: Let's think step by step and output the final answer.
    """
    # Insert question into template
    prompt = custom_prompt_template.format(question=data['question'])

    # Generate and display
    print(f"Question: {data['question']}\n")
    print(f"Ground Truth: {data['answer']}\n")
    
    response = utils.generate_response(model, tokenizer, prompt)
    print(f"Model Answer: {response}\n")
    print(f"Extracted Number: {utils.extract_number(response)}")

if __name__ == "__main__":
    main()
