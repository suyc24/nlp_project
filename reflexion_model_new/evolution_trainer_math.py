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
            max_model_len=4096,
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
        # 抽象参数 (Greedy Decode)
        self.params_abstract = SamplingParams(
            temperature=0.0, top_p=0.9, max_tokens=128,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        print("   -> Loading Embedder (Force CPU to save GPU memory)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu", cache_folder=HF_CACHE_DIR)       
        self.memory = MemoryManager(reset=True)
        self.debug_log_path = "debug_trace.jsonl"
        open(self.debug_log_path, "w").close() 

    def get_hash(self, text):
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
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

    def construct_abstraction_prompt(self, q):
        content = f"""
            Task: Identify the core mathematical concept and intent of the following problem.
            Output a concise, abstract description of the problem and its condition.
            ### Requirements
            - **Format**: Your output must be a single sentence following this pattern: "[Abstract Problem Type] given that [Specific Conditions from the Question including numerical constraints, relationships, and constraints]"
            - **Strict Constraint**: Do NOT include any specific numbers (e.g., 16, 3) or specific nouns (e.g., eggs, ducks) from the current problem. The principle must be universal. 

            [Example]
            Q: John has 5 apples and buys 3 more. How many?
            A: Calculating the total sum of objects given that each part is provided.

            [Target]
            Q: {q}
            A:"""
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    def verify_step_vllm(self, partial_prompt, gt, k=1):
        """
        验证当前步骤是否正确。
        Prompt 构造逻辑：给定前面的步骤，询问最后一步是否与 Ground Truth 矛盾。
        """
        # 构造验证 Prompt
        verify_content = f"""
        I am solving a math problem.
        [Previous Steps]
        {partial_prompt.replace('<|im_start|>user', '').replace('<|im_start|>assistant', '').strip()}

        [Ground Truth Answer]
        {gt}

        Evaluate ONLY the last step provided above.
        Is this step logically correct and consistent with leading to the Ground Truth?
        Answer strictly "Yes" or "No".
        """
        full_prompt = f"<|im_start|>user\n{verify_content}<|im_end|>\n<|im_start|>assistant\n"
        
        # 这里的 k 参数如果是为了多次采样验证，可以在这里扩展，目前简化为一次
        outputs = self.batch_generate_vllm([full_prompt], self.params_verify)
        resp = outputs[0].lower()
        
        return "yes" in resp and "no" not in resp

    def self_explore_phase(self, epoch, still_incorrect_indices, chunk_questions, chunk_answers, index_to_rules=None):
        """
        [极速版] Self-Explore 机制：
        1. Hindsight: 强制生成正确路径 (Batch)
        2. Contrast: 对比生成规则 (Batch)
        3. Verification: 收集所有候选规则一次性并行验证 (Batch)
        """
        if not still_incorrect_indices:
            return

        tqdm.write(f" 🔍 Self-Exploring {len(still_incorrect_indices)} samples (Batch Optimized)...")

        # 1. 准备数据
        target_questions = [chunk_questions[i] for i in still_incorrect_indices]
        target_answers = [chunk_answers[i] for i in still_incorrect_indices]

        # ==========================================================
        # Step 1: 错误路径 & 正确路径
        # ==========================================================
        prompts_wrong = [self.construct_prompt(q) for q in target_questions]
        traces_wrong = self.batch_generate_vllm(prompts_wrong, self.params_inference)

        prompts_correct = []
        for q, gt in zip(target_questions, target_answers):
            hindsight_prompt = (
                f"Question: {q}\n"
                f"The correct answer is known to be: {gt}.\n"
                f"Please provide a correct, step-by-step mathematical derivation that results in this answer.\n"
                f"Answer step-by-step:"
            )
            prompts_correct.append(f"<|im_start|>user\n{hindsight_prompt}<|im_end|>\n<|im_start|>assistant\n")

        traces_correct = self.batch_generate_vllm(prompts_correct, self.params_inference)

        # ==========================================================
        # Step 2: Contrast 反思 (Two-Step Generation)
        # ==========================================================
        valid_indices = []
        
        # 1. Filter valid candidates
        for i in range(len(target_questions)):
            gt = target_answers[i]
            w_trace = traces_wrong[i]
            c_trace = traces_correct[i]
            if not self.check_answer(w_trace, gt) and self.check_answer(c_trace, gt):
                valid_indices.append(i)
        
        if not valid_indices:
            return

        # 2. Step 2a: Abstract Trigger (Greedy)
        # Use the exact same prompt and params as retrieval to ensure consistency
        abstract_prompts = [self.construct_abstraction_prompt(target_questions[i]) for i in valid_indices]
        abstract_outputs = self.batch_generate_vllm(abstract_prompts, self.params_abstract)

        # 3. Step 2b: Generate Strategy (High Temp)
        strategy_prompts = []
        for idx, i in enumerate(valid_indices):
            q = target_questions[i]
            c_trace = traces_correct[i]
            trigger = abstract_outputs[idx]
            
            strategy_content = f"""
[TASK] Define a general mathematical strategy to solve this type of problem, based on the identified abstract condition and the correct solution.

[Question]
{q}

[Correct Solution]
{c_trace}

[Identified Abstract Condition (Trigger)]
{trigger}

STRICT OUTPUT FORMAT INSTRUCTION:
Output the Strategy (B) that corresponds to the Trigger (A).

[Format]
**Strategy (B)**: [General Method/Principle]

[!!FORBIDDEN!!]
1. **NO NUMBERS**: Absolutely NO digits or number words.
2. **NO SPECIFIC NOUNS**: Use abstract terms like "objects", "values".
3. **LENGTH LIMIT**: MUST be under 20 words.
[YOUR TURN]
            """
            strategy_prompts.append(f"<|im_start|>user\n{strategy_content}<|im_end|>\n<|im_start|>assistant\n")

        strategy_outputs = self.batch_generate_vllm(strategy_prompts, self.params_reflection)

        # 4. Combine for next step
        reflections = []
        contrast_metadata_map = valid_indices 
        
        for idx, strategy_text in enumerate(strategy_outputs):
            trigger_text = abstract_outputs[idx]
            # Synthesize the text for parse_reflection
            combined_text = f"**Trigger (A)**: {trigger_text}\n{strategy_text}"
            reflections.append(combined_text)

        # ==========================================================
        # Step 3: 解析规则 & 构造验证 prompt（🔧 FIX：完整 metadata）
        # ==========================================================
        verify_prompts = []
        verify_candidates_metadata = []

        for idx, reflection_text in enumerate(reflections):
            original_idx = contrast_metadata_map[idx]

            q = target_questions[original_idx]
            gt = target_answers[original_idx]

            parsed = self.parse_reflection(reflection_text)
            if not parsed:
                continue

            trigger, strategy = parsed
            if self.has_specific_numbers(strategy, q):
                continue
            if len(strategy) < 10:
                continue

            temp_embed = self.embedder.encode(trigger + " " + strategy, convert_to_numpy=True)
            try:
                existing = self.memory.collection.query(query_embeddings=[temp_embed], n_results=1)
                if existing['ids'] and existing['ids'][0]:
                    # 如果相似度非常高 (distance < 0.2)，说明库里已经有了，直接跳过，省下验证的时间
                    if existing['distances'][0][0] < 0.2:
                        print(f"   Duplicate rule detected (dist={existing['distances'][0][0]:.3f}), skipping verification.")
                        continue
            except Exception as e:
                pass

            v_prompt = self.construct_prompt(q, context=strategy)
            verify_prompts.append(v_prompt)

            # 🔧 FIX：保存 question / index / gt，避免错位
            verify_candidates_metadata.append({
                "trigger": trigger,
                "strategy": strategy,
                "question": q,
                "gt": gt,
                "local_idx": original_idx
            })

        # ==========================================================
        # Step 4: Batch 验证
        # ==========================================================
        if not verify_prompts:
            return

        verify_outputs = self.batch_generate_vllm(verify_prompts, self.params_verify)

        new_patterns = []
        new_strategies = []
        new_embeddings_inputs = []
        new_source_hashes = []

        for i, pred in enumerate(verify_outputs):
            meta = verify_candidates_metadata[i]
            gt = meta["gt"]
            q = meta["question"]

            if self.check_answer(pred, gt):
                self.log_debug({
                    "epoch": epoch,
                    "phase": "rule_verified",
                    "trigger": meta["trigger"],
                    "strategy": meta["strategy"],
                    "question": q,   # ✅ 绝对正确
                    "gt": gt
                })

                tqdm.write(f"    ✅ Rule Verified: {meta['trigger'][:40]}... -> {meta['strategy'][:40]}...")
                new_patterns.append(meta["trigger"])
                new_strategies.append(meta["strategy"])
                new_embeddings_inputs.append(meta["trigger"])
                new_source_hashes.append(self.get_hash(q))

        if new_patterns:
            print(f" 💾 Adding {len(new_patterns)} high-quality rules to memory...")
            embeddings = self.embedder.encode(new_embeddings_inputs, convert_to_numpy=True)
            self.memory.add_experience_batch(new_patterns, new_strategies, embeddings, source_q_hashes=new_source_hashes)


    def run_full_evolution(self):
        # 1. 准备数据
        # dataset = load_dataset("gsm8k", "main")['train']
        full_dataset = load_dataset("qwedsacf/competition_math", split="train")
        
        # Split 10% for testing
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        dataset = split_dataset['train']
        
        total_len = len(dataset)
        print(f"⚡️ 正在预计算 {total_len} 条问题的 Embedding (CPU Mode)...")
        
        all_questions_raw = dataset['problem']
        all_answers_raw = dataset['solution']
        
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
                    
                    # Prepare hashes for the questions that actually used RAG
                    final_q_hashes = [self.get_hash(wrong_questions[k]) for k in range(len(wrong_questions)) if rag_prompts[k] is not None]

                    if final_usage:
                        self.memory.update_scores_batch(final_usage, rag_is_correct_for_update, rag_outputs_for_update, current_q_hashes=final_q_hashes)

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
        
        print("🏁 Training Finished. Pruning probationary rules...")
        self.memory.prune_probationary_rules()

    def batch_abstract_for_retrieval(self, questions):
        """
        将具体问题转化为抽象的数学考点描述，用于检索。
        """
        prompts = [self.construct_abstraction_prompt(q) for q in questions]
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
        elif "\\boxed{" in gt:
            try:
                start = gt.rfind("\\boxed{") + 7
                count = 1
                end = start
                while count > 0 and end < len(gt):
                    if gt[end] == '{': count += 1
                    elif gt[end] == '}': count -= 1
                    end += 1
                gold = self.extract_number(gt[start:end-1])
            except:
                gold = self.extract_number(gt)
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
