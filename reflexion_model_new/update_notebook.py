import json
import sys

file_path = '/root/autodl-tmp/nlp_project/reflexion_model/test.ipynb'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Find the cell
target_cell = None
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_text = "".join(cell['source'])
        if 'class ReflexionTrainerFull' in source_text and 'def self_explore_phase' in source_text:
            target_cell = cell
            break

if target_cell:
    source = "".join(target_cell['source'])
    
    # We need to be careful with exact matching. 
    # I will identify the start and end of the block to replace.
    
    start_marker = "# ==========================================================\n        # Step 1: 错误路径 & 正确路径\n        # =========================================================="
    end_marker = "traces_correct = self.batch_generate_vllm(prompts_correct, self.params_inference)"
    
    start_idx = source.find(start_marker)
    end_idx = source.find(end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Include the end_marker in the replacement range?
        # The end_marker is the last line of the block I want to replace.
        # So I should replace up to end_idx + len(end_marker)
        
        replace_end_idx = end_idx + len(end_marker)
        
        new_block = """# ==========================================================
        # Step 1: 错误路径 & 正确路径
        # ==========================================================
        prompts_wrong = [self.construct_prompt(q) for q in target_questions]
        traces_wrong = self.batch_generate_vllm(prompts_wrong, self.params_inference)

        # --- 探索多种 Prompt 以获得正确路径 ---
        traces_correct = [""] * len(target_questions)
        solved_mask = [False] * len(target_questions)
        
        exploration_templates = [
            # Template 1: Standard Hindsight
            lambda q, gt: (
                f"Question: {q}\\n"
                f"The correct answer is known to be: {gt}.\\n"
                f"Please provide a correct, step-by-step mathematical derivation that results in this answer.\\n"
                f"Answer step-by-step:"
            ),
            # Template 2: Roleplay Expert
            lambda q, gt: (
                f"You are a mathematics professor.\\n"
                f"Problem: {q}\\n"
                f"Target Answer: {gt}\\n"
                f"Explain the solution clearly and logically to a student.\\n"
                f"Solution:"
            ),
             # Template 3: Step-by-step verification
            lambda q, gt: (
                f"Question: {q}\\n"
                f"Final Answer: {gt}\\n"
                f"Please think step by step to verify why this answer is correct.\\n"
                f"Verification:"
            )
        ]

        for template_fn in exploration_templates:
            # 找出还未生成正确路径的索引
            remaining_indices = [i for i, solved in enumerate(solved_mask) if not solved]
            if not remaining_indices:
                break
            
            current_prompts = []
            for i in remaining_indices:
                prompt_content = template_fn(target_questions[i], target_answers[i])
                current_prompts.append(f"<|im_start|>user\\n{prompt_content}<|im_end|>\\n<|im_start|>assistant\\n")
            
            # Batch 生成
            current_outputs = self.batch_generate_vllm(current_prompts, self.params_inference)
            
            # 检查并更新
            for idx_in_batch, output in enumerate(current_outputs):
                original_idx = remaining_indices[idx_in_batch]
                if self.check_answer(output, target_answers[original_idx]):
                    traces_correct[original_idx] = output
                    solved_mask[original_idx] = True
        
        # 统计找回率
        success_count = sum(solved_mask)
        tqdm.write(f"    ✨ Correct path recovery: {success_count}/{len(target_questions)}")"""
        
        new_source = source[:start_idx] + new_block + source[replace_end_idx:]
        
        # Convert back to list of lines
        # We need to split by \n but keep the \n
        lines = new_source.split('\n')
        # Add \n back to all lines except maybe the last one if it didn't have one
        # Actually, split removes the delimiter.
        
        new_source_list = [line + '\n' for line in lines[:-1]]
        new_source_list.append(lines[-1]) # Last line might not need \n or might be empty
        
        target_cell['source'] = new_source_list
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Successfully updated notebook.")
        
    else:
        print("Could not find the code block markers.")
        print(f"Start found: {start_idx}, End found: {end_idx}")
        # Debug
        if start_idx == -1:
            print("Start marker not found in source.")
        if end_idx == -1:
            print("End marker not found in source.")

else:
    print("Could not find the target cell.")
