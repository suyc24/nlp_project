from config import * 
import re
import torch
import json
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from principle_manager import MemoryManager

# ================= vLLM 进化训练器 (主要修改区域) =================
class ReflexionTrainerFull:
    def __init__(self):
        print("🚀 初始化训练环境 (vLLM Accelerated Mode)...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            max_model_len=2048,
            download_dir=HF_CACHE_DIR
        )
        
        # 标准推理参数
        self.params_inference = SamplingParams(
            temperature=0.2, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        # 反思参数 (高创造性)
        self.params_reflection = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        # 验证参数 (低容错)
        self.params_verify = SamplingParams(
            temperature=0.1, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        # 抽象参数 (低长度)
        self.params_abstract = SamplingParams(
            temperature=0.5, top_p=0.9, max_tokens=64,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        print("   -> Loading Embedder (Force CPU to save GPU memory)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu", cache_folder=HF_CACHE_DIR)       
        self.memory = MemoryManager(reset=True)
        self.debug_log_path = "debug_trace.jsonl"
        open(self.debug_log_path, "w").close() 
        
    def log_debug(self, data):
        import json
        with open(self.debug_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def batch_generate_vllm(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def has_specific_numbers(self, text, question_text):
        q_nums = set(re.findall(r'\b\d+\b', question_text))
        r_nums = set(re.findall(r'\b\d+\b', text))
        whitelist = {'1', '2', '3', '4', '10', '100', '180', '360'} 
        for n in r_nums:
            if n in q_nums and n not in whitelist:
                return True 
        return False

    def clean_rule_text(self, text):
        text = text.replace("```markdown", "").replace("```", "").strip()
        patterns_to_remove = [
            r"^Sure,.*?\n", r"^Here is.*?\n", r"^Certainly.*?\n",
            r"^To summarize.*?\n", r"You are an AI assistant.*?",
            r"Assistant:.*?", r"Great job!.*?", r"#### \d+"
        ]
        for pat in patterns_to_remove:
            text = re.sub(pat, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
        if text.count("**Trigger (A)**") > 1:
            first_occurrence = text.find("**Trigger (A)**", text.find("**Trigger (A)**") + 1)
            text = text[:first_occurrence]
        if text.count("**Strategy (B)**") > 1:
            first_occurrence = text.find("**Strategy (B)**", text.find("**Strategy (B)**") + 1)
            text = text[:first_occurrence]
        return text.strip()

    def parse_reflection(self, text):
        cleaned_text = self.clean_rule_text(text)
        trigger_match = re.search(r"(?:\*\*Trigger \(A\)\*\*|Trigger \(A\)|Trigger):?\s*(.*?)(?=\n|(?:\*\*Strategy)|$)", cleaned_text, re.DOTALL | re.IGNORECASE)
        strategy_match = re.search(r"(?:\*\*Strategy \(B\)\*\*|Strategy \(B\)|Strategy):?\s*(.*?)(?=\n\s*(?:\*\*Trigger)|$)", cleaned_text, re.DOTALL | re.IGNORECASE)
        
        if trigger_match and strategy_match:
            t_text = trigger_match.group(1).strip()
            s_text = strategy_match.group(1).strip()
            t_text = re.sub(r"\s+", " ", t_text)
            s_text = re.sub(r"\s+", " ", s_text)

            if len(t_text) < 5 or len(s_text) < 5: return None
            if len(t_text) > 300: t_text = t_text[:300] 
            if "Strategy" in t_text: t_text = t_text.split("Strategy")[0].strip()
            return t_text, s_text
        return None

    def construct_prompt(self, q, context=None):
        if context:
            content = f"""
                [Reference Rules]
                {context}

                [Question]
                {q}

                [Instruction]
                1. Read the Reference Rules carefully.
                2. First, decide which rule is relevant to the question.
                3. If a rule is relevant, write "Selected Rule: [Rule Content]".
                4. If no rule is relevant, write "No suitable rule found".
                5. Then, solve the problem step-by-step using the selected rule (if any).

                Answer:"""
        else:
            content = f"Question: {q}\nAnswer step-by-step:"
            
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def verify_step_vllm(self, partial_prompt, gt, k=5):
        """
        验证当前步骤是否正确 (Monte Carlo Rollout).
        逻辑：从当前步骤继续生成 k 次，如果其中有一次能得到正确答案，则认为当前步骤有效。
        """
        # 使用较高的 temperature 进行探索
        params_rollout = SamplingParams(
            n=k, 
            temperature=0.7, 
            top_p=0.9, 
            max_tokens=1024, 
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        
        # vLLM generate
        outputs = self.llm.generate([partial_prompt], params_rollout, use_tqdm=False)
        generated_texts = [o.text for o in outputs[0].outputs]
        
        for text in generated_texts:
            # 检查是否包含正确答案
            # 1. 优先尝试提取 boxed
            matches = re.findall(r'\\boxed\{([^}]+)\}', text)
            if matches:
                pred_val = matches[-1]
                if self.check_answer(str(pred_val), gt):
                    return True
            
            # 2. Fallback: 直接检查整个文本中的最后一个数字
            if self.check_answer(text, gt):
                return True
                
        return False

    def self_explore_phase(self, epoch, still_incorrect_indices, chunk_questions, chunk_answers, index_to_rules=None):
        """
        [Updated] Self-Explore Mechanism based on Iterative Correction (test.ipynb logic).
        """
        if not still_incorrect_indices:
            return

        from transformers import AutoTokenizer
        # Initialize tokenizer if not already done
        if not hasattr(self, 'tokenizer'):
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

        tqdm.write(f" 🔍 Self-Exploring {len(still_incorrect_indices)} samples (Iterative Correction)...")

        # 1. Prepare Data
        target_questions = [chunk_questions[i] for i in still_incorrect_indices]
        target_answers = [chunk_answers[i] for i in still_incorrect_indices]

        # Helper for answer extraction
        def extract_answer_boxed(text):
            matches = re.findall(r'\\boxed\{([^}]+)\}', text)
            if matches:
                return matches[-1]
            return str(self.extract_number(text))

        # Principle Generation Prompt Template
        generate_principle_prompt =  """You are a helpful math assistant. 

### Question
{question}

### Student's Partial Reasoning
{partial_reasoning}

### The Error Step
{error_step}

### Right Answer
{answer}

### Instruction
The student failed to solve the problem correctly due to the error in the step above. 
Please provide a **generalized principle** to help the student understand the underlying concept and avoid similar mistakes.

### Steps to Generate the Principle
1. **Abstract the Problem and its Condition**: Identify the general category of this problem (e.g., "calculating compound probabilities", "finding the remaining amount after multiple subtractions") without referring to specific numbers or items in this problem.
2. **Identify the Method**: Determine the correct general method or logical step required to solve this type of problem.
3. **Formulate the Hint**: Combine these into a single sentence.

### Requirements
- **Format**: Your output must be a single sentence following this pattern: "To solve [Abstract Problem Description] with condition that [condition and specialty of the Problem], consider [General Method/Principle]."
- **Strict Constraint**: Do NOT include any specific numbers (e.g., 16, 3) or specific nouns (e.g., eggs, ducks) from the current problem. The principle must be universal. """


        new_patterns = []
        new_strategies = []
        new_embeddings_inputs = []
        
        MAX_RETRIES = 3

        for idx, (q, gt) in enumerate(zip(target_questions, target_answers)):
            # Initial Prompt
            messages = [{"role": "user", "content": f"Question: {q.strip()}\nLet's solve this step by step. At the end, please enclose the final answer in \\boxed{{}}."}]
            current_question_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            accumulated_principles = []
            
            for attempt in range(MAX_RETRIES + 1):
                # Construct Prompt with Hints
                if accumulated_principles:
                    hints_str = "\n".join([f"Hint {i+1}: {p}" for i, p in enumerate(accumulated_principles)])
                    content = f"Here are some hints to help you solve the problem:\n{hints_str}\n\nQuestion: {q.strip()}\nLet's solve this step by step. At the end, please enclose the final answer in \\boxed{{}}."
                    messages = [{"role": "user", "content": content}]
                    current_input_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    current_input_prompt = current_question_prompt

                # Generate CoT
                completions = self.batch_generate_vllm([current_input_prompt], self.params_inference)
                comp = completions[0]
                
                # Check correctness
                pred_ans = extract_answer_boxed(comp)
                is_correct = False
                if pred_ans:
                    is_correct = self.check_answer(str(pred_ans), gt)
                
                if is_correct:
                    # If solved and we have principles, collect them
                    if accumulated_principles:
                        for p in accumulated_principles:
                            # Try to parse trigger
                            match = re.search(r"To solve (.*?), consider (.*)", p, re.IGNORECASE)
                            if match:
                                trigger_text = match.group(1).strip()
                                new_patterns.append(trigger_text)
                                new_strategies.append(p)
                                new_embeddings_inputs.append(trigger_text)
                            else:
                                new_patterns.append(q) # Fallback trigger
                                new_strategies.append(p)
                                new_embeddings_inputs.append(q)
                    break # Solved, move to next question
                
                # If incorrect, analyze
                if attempt < MAX_RETRIES:
                    steps = [s.strip() for s in comp.split('\n') if s.strip()]
                    verify_base_prompt = current_input_prompt
                    found_error = False
                    
                    for i, step in enumerate(steps):
                        verify_base_prompt += step + "\n"
                        # Verify Step
                        if not self.verify_step_vllm(verify_base_prompt, gt):
                            found_error = True
                            error_step = step
                            partial_reasoning = "\n".join(steps[:i]) if i > 0 else "(No previous steps)"
                            
                            # Generate Principle
                            short_gt = gt.split("####")[1].strip() if "####" in gt else gt
                            p_prompt_content = generate_principle_prompt.format(
                                question=q,
                                partial_reasoning=partial_reasoning,
                                error_step=error_step,
                                answer=short_gt
                            )
                            p_messages = [
                                {"role": "system", "content": "You are a helpful math assistant."},
                                {"role": "user", "content": p_prompt_content}
                            ]
                            p_full_prompt = self.tokenizer.apply_chat_template(p_messages, tokenize=False, add_generation_prompt=True)
                            
                            p_outputs = self.batch_generate_vllm([p_full_prompt], self.params_reflection)
                            principle = p_outputs[0]
                            accumulated_principles.append(principle)
                            break # Stop verifying, retry
                    
                    if not found_error:
                        pass # Could not find error, just retry loop continues

        if new_patterns:
            print(f" 💾 Adding {len(new_patterns)} high-quality rules to memory...")
            embeddings = self.embedder.encode(new_embeddings_inputs, convert_to_numpy=True)
            self.memory.add_experience_batch(new_patterns, new_strategies, embeddings)


    def run_full_evolution(self):
        # 1. 准备数据
        dataset = load_dataset("gsm8k", "main")['train'] #.select(range(200)) 
        
        total_len = len(dataset)
        print(f"⚡️ 正在预计算 {total_len} 条问题的 Embedding (CPU Mode)...")
        
        all_questions_raw = dataset['question']
        all_answers_raw = dataset['answer']
        
        # 预计算 Embeddings (Epoch 间复用)
        all_q_embeddings = self.embedder.encode(all_questions_raw, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        
        print(f"🔥 开始全量进化训练 (vLLM Speedup) - 目标准确率: {TARGET_ACCURACY}%")

        epoch = 0
        best_acc = 0.0

        while best_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS:
            epoch += 1
            print(f"\n======== Epoch {epoch}/{MAX_EPOCHS} ========")
            
            # 1. Shuffle 数据索引
            indices = np.random.permutation(total_len)
            
            epoch_correct_count = 0
            epoch_total_count = 0
            
            pbar = tqdm(range(0, total_len, CHUNK_SIZE), desc=f"Epoch {epoch} Training")
            
            for chunk_start in pbar:
                chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
                current_batch_indices = indices[chunk_start:chunk_end]
                
                # 获取当前 Batch 的原始数据
                chunk_questions = [all_questions_raw[i] for i in current_batch_indices]
                chunk_answers = [all_answers_raw[i] for i in current_batch_indices]
                chunk_q_embeddings = all_q_embeddings[current_batch_indices]
                
                # ================= 阶段 1: Zero-shot 推理 =================
                zero_shot_prompts = [self.construct_prompt(q) for q in chunk_questions]
                zs_outputs = self.batch_generate_vllm(zero_shot_prompts, self.params_inference)
                
                # 评估 Zero-shot 结果
                zs_is_correct = []
                incorrect_local_indices = [] # 记录在这个 chunk 中做错的下标
                
                for idx, pred in enumerate(zs_outputs):
                    gt = chunk_answers[idx]
                    is_right = self.check_answer(pred, gt)
                    zs_is_correct.append(is_right)

                    if not is_right:
                        incorrect_local_indices.append(idx)
                    epoch_total_count += 1

                chunk_final_correct = zs_is_correct[:]
                
                # ================= 阶段 2: RAG 重算 (仅针对错题) =================
                rag_usage_for_update = []
                rag_is_correct_for_update = []
                rag_outputs_for_update = []              
                still_incorrect_indices = []   # RAG 后依然做错的下标，用于 Self-Explore
                
                # 用于记录每道错题用了什么规则，传给 Self-Explore 打印日志
                rag_rules_map = {} 

                if incorrect_local_indices:
                    # 1. 准备错题数据
                    wrong_questions = [chunk_questions[i] for i in incorrect_local_indices]
                    wrong_embeddings = chunk_q_embeddings[incorrect_local_indices]
                    wrong_answers = [chunk_answers[i] for i in incorrect_local_indices]
                    
                    # 2. 检索规则
                    print(f"    🧠 Abstracting {len(wrong_questions)} questions for better retrieval...")
                    abstract_intent_list = self.batch_abstract_for_retrieval(wrong_questions)

                    # 2. 对“抽象描述”进行 Embedding，而不是对原题 Embedding
                    query_embeddings = self.embedder.encode(abstract_intent_list, convert_to_numpy=True).tolist()
                    
                    # 3. 使用抽象 Embedding 进行检索
                    retrieved_batch = self.memory.batch_retrieve(query_embeddings, top_k=3, threshold=0.6)
                    
                    rag_prompts = []
                    
                    for k, q in enumerate(wrong_questions):
                        original_idx = incorrect_local_indices[k]
                        rules_list = retrieved_batch[k]
                        
                        if rules_list:
                            context_text = "\n".join([f"[Rule {i+1}]: {r[0]}" for i, r in enumerate(rules_list)])
                            rag_prompts.append(self.construct_prompt(q, context_text))
                            rag_usage_for_update.append(rules_list)
                            rag_rules_map[original_idx] = [r[0] for r in rules_list]
                        else:
                            # 如果没检索到规则，就没必要 RAG 重算了
                            rag_usage_for_update.append([]) 
                            rag_prompts.append(None) 
                            still_incorrect_indices.append(original_idx)

                    # 3. 执行 RAG 推理 (只推理有 Prompts 的部分)
                    real_rag_prompts = [p for p in rag_prompts if p is not None]
                    if real_rag_prompts:
                        real_rag_outputs = self.batch_generate_vllm(real_rag_prompts, self.params_inference)
                    
                    # 4. 评估 RAG 结果并准备 Update 数据
                    output_cursor = 0

                    for k, q in enumerate(wrong_questions):
                        original_idx = incorrect_local_indices[k]
                        
                        if rag_prompts[k] is None:
                            continue
                            
                        pred = real_rag_outputs[output_cursor]
                        output_cursor += 1
                        
                        gt = wrong_answers[k]
                        is_right = self.check_answer(pred, gt)
                        
                        # 记录 Update 数据
                        rag_is_correct_for_update.append(is_right)
                        rag_outputs_for_update.append(pred)
                        
                        self.log_debug({
                            "epoch": epoch,
                            "phase": "rag",
                            "question": q,
                            "used_rules": rag_rules_map.get(original_idx, []),
                            "pred": pred,
                            "gt": gt,
                            "is_correct": is_right
                        })

                        if is_right:
                            chunk_final_correct[original_idx] = True
                        else:
                            still_incorrect_indices.append(original_idx)
                    
                    # 5. 更新 Memory 分数
                    final_usage = [rag_usage_for_update[k] for k in range(len(wrong_questions)) if rag_prompts[k] is not None]
                    
                    if final_usage:
                        self.memory.update_scores_batch(final_usage, rag_is_correct_for_update, rag_outputs_for_update)

                # ================= 阶段 3: Self-Explore (仅针对 RAG 后依然错误的题) =================
                if still_incorrect_indices:
                    self.self_explore_phase(
                        epoch,
                        still_incorrect_indices, 
                        chunk_questions, 
                        chunk_answers, 
                        index_to_rules=rag_rules_map
                    )

                # --- 定期淘汰 ---
                if (chunk_start // CHUNK_SIZE) % 5 == 0:
                    self.memory.prune_db(threshold=0.25)

                epoch_correct_count += sum(chunk_final_correct)

                # 更新进度条信息 (ZS+RAG Accuracy)
                batch_acc = sum(chunk_final_correct) / len(chunk_questions) * 100
                pbar.set_postfix({"Acc(ZS+RAG)": f"{batch_acc:.1f}%", "DB": self.memory.collection.count()})
            
            # --- Epoch 总结 ---
            current_epoch_acc = (epoch_correct_count / epoch_total_count) * 100
            print(f"\n📊 Epoch {epoch} 完成 | 综合准确率 (ZS+RAG): {current_epoch_acc:.2f}% (Target: {TARGET_ACCURACY}%)")
            
            if current_epoch_acc > best_acc:
                best_acc = current_epoch_acc
            
            if current_epoch_acc >= TARGET_ACCURACY:
                print(f"🎉 恭喜！达到目标准确率 ({current_epoch_acc:.2f}%)，训练完成！")
                break
            elif epoch >= MAX_EPOCHS:
                print("⚠️ 达到最大 Epoch 限制，停止训练。")
                break
            else:
                print("🔄 表现未达标，继续下一轮训练...")

    def batch_abstract_for_retrieval(self, questions):
        """
        将具体问题转化为抽象的数学考点描述，用于检索。
        """
        prompts = []
        for q in questions:
            content = f"""
            Task: Identify the core mathematical concept and intent of the following problem.
            Output a concise, abstract description of the problem and its condition.
            ### Requirements
            - **Format**: Your output must be a single sentence following this pattern: "[Abstract Problem Description] with condition that [condition and specialty of the Problem]"
            - **Strict Constraint**: Do NOT include any specific numbers (e.g., 16, 3) or specific nouns (e.g., eggs, ducks) from the current problem. The principle must be universal. 

            [Example]
            Q: John has 5 apples and buys 3 more. How many?
            A: Calculating the total sum of objects with condition that the each part is provided.

            [Target]
            Q: {q}
            A:"""
            prompts.append(f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n")
        
        abstract_queries = self.batch_generate_vllm(prompts, self.params_abstract)
        
        return abstract_queries

    def extract_number(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    def check_answer(self, pred, gt):
        if "####" in gt:
            gold = self.extract_number(gt.split("####")[1])
        else:
            gold = self.extract_number(gt)
        pred_num = self.extract_number(pred)
        if gold is None or pred_num is None: return False
        return abs(gold - pred_num) < 1e-4

    def cleanup(self):
        print("🧹 正在清理显存和 vLLM 进程...")
        if hasattr(self, 'llm'):
            del self.llm
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        try:
            import ray
            if ray.is_initialized():
                ray.shutdown()
        except ImportError:
            pass
        print("✅ 清理完成！")

if __name__ == "__main__":
    trainer = None
    try:
        trainer = ReflexionTrainerFull()
        trainer.run_full_evolution()
    except KeyboardInterrupt:
        print("\n🛑 用户强制停止训练")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        raise e
    finally:
        if trainer is not None:
            trainer.cleanup()