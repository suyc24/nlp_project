import os
HF_CACHE_DIR = "/root/autodl-tmp/hf_cache"
os.makedirs(HF_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HUGGINGFACE_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import shutil
import time
import random
import re
import torch
import chromadb
import json
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams
from tqdm import tqdm

# ================= Configuration Area =================
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
DB_PATH = "./reflexion_full_db"
CHUNK_SIZE = 64  # processing unit size
MAX_NEW_TOKENS = 256
GPU_MEMORY_UTILIZATION = 0.90 
TARGET_ACCURACY = 92.0  # target accuracy percentage - stop when reached
MAX_EPOCHS = 5        # maximum training epochs to avoid infinite loops
IF_SELF_EXPLORE = False

GENERATE_PRINCIPLE_PROMPT = """You are a helpful math assistant. 

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
1. **Abstract the Problem**: Identify the general category of this problem (e.g., "calculating compound probabilities", "finding the remaining amount after multiple subtractions") without referring to specific numbers or items in this problem.
2. **Identify the Method**: Determine the correct general method or logical step required to solve this type of problem.
3. **Formulate the Hint**: Combine these into a single sentence.

### Requirements
- **Format**: Your output must be a single sentence following this pattern: "To solve [Abstract Problem Description], consider [General Method/Principle]."
- **Strict Constraint**: Do NOT include any specific numbers (e.g., 16, 3) or specific nouns (e.g., eggs, ducks) from the current problem. The principle must be universal. """

# ================= 1. Memory Manager (unchanged behavior) =================
class MemoryManager:
    # Detailed: Initialize MemoryManager.
    # Parameters:
    # - reset (bool): If True and the DB path exists, the persistent DB directory will be removed to start fresh.
    # Implementation details: removes the DB directory when requested, creates a Chromadb PersistentClient at DB_PATH,
    # creates/gets the 'rule_book' collection, and initializes in-memory statistics structures used for tracking rule quality.
    def __init__(self, reset=False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        # In-memory stats
        self.skill_stats = {} 
        self.current_step = 0 # global counter

    # Detailed: Retrieve nearest rules for a batch of query embeddings.
    # Parameters:
    # - query_embeddings (Sequence[ndarray|list]): A list/array of embeddings to query against the DB.
    # - top_k (int): Maximum number of nearest neighbors to return per query.
    # - threshold (float): Distance threshold to filter out weak matches.
    # Returns:
    # - A list of lists, one per query embedding. Each inner list contains tuples of (document_text, distance, id).
    # Implementation details: queries the chromadb collection with the provided embeddings, filters results by distance,
    # and handles empty DB or query errors by returning empty lists aligned to the input length.
    def batch_retrieve(self, query_embeddings, top_k=3, threshold=0.5):
        count = self.collection.count()
        if count == 0: return [[] for _ in range(len(query_embeddings))]
        
        real_k = min(top_k, count)
        results_list = [] 
        try:
            results = self.collection.query(query_embeddings=query_embeddings, n_results=real_k)
            for i in range(len(query_embeddings)):
                sample_rules = []
                if results['ids'][i]:
                    for j in range(len(results['ids'][i])):
                        dist = results['distances'][i][j]
                        if dist < threshold:
                            doc = results['documents'][i][j]
                            sid = results['ids'][i][j]
                            sample_rules.append((doc, dist, sid))
                results_list.append(sample_rules)
        except Exception as e:
            print(f"Retrieval Error: {e}")
            return [[] for _ in range(len(query_embeddings))]
        return results_list

    # Detailed: Add new rule experiences into the memory DB in batch.
    # Parameters:
    # - patterns_A (list[str]): List of trigger pattern texts (Trigger A) extracted from reflections.
    # - strategies_B (list[str]): Corresponding strategy texts (Strategy B) to store as documents.
    # - embeddings_A (list[array-like]): Corresponding embeddings for the patterns, used for vector search.
    # Implementation details: performs a nearest-neighbor check to avoid near-duplicate entries (distance < 0.15).
    # If a near-duplicate exists, it boosts that rule's score; otherwise it creates new unique ids, stores documents
    # and embeddings into chromadb and initializes their tracking stats in `self.skill_stats`.
    def add_experience_batch(self, patterns_A, strategies_B, embeddings_A):
        if not patterns_A: return
        unique_patterns = []
        unique_strategies = []
        unique_embeddings = []
        
        for i, emb in enumerate(embeddings_A):
            try:
                existing = self.collection.query(query_embeddings=[emb], n_results=1)
                if existing['ids'] and existing['ids'][0] and existing['distances'][0][0] < 0.15: 
                    exist_id = existing['ids'][0][0]
                    if exist_id in self.skill_stats:
                        self.skill_stats[exist_id]['score'] = min(1.0, self.skill_stats[exist_id]['score'] + 0.05)
                    continue 
            except:
                pass
            unique_patterns.append(patterns_A[i])
            unique_strategies.append(strategies_B[i])
            unique_embeddings.append(embeddings_A[i])

        if not unique_patterns: return

        new_ids = [f"rule_{int(time.time())}_{i}_{random.randint(0,999)}" for i in range(len(unique_patterns))]
        metadatas = [{"pattern": p} for p in unique_patterns]
        
        self.collection.add(
            ids=new_ids,
            embeddings=unique_embeddings,
            documents=unique_strategies,
            metadatas=metadatas
        )
        
        for sid in new_ids:
            self.skill_stats[sid] = {
                "score": 0.5, "usage": 0, "history_correct": 0, 
                "created_step": self.current_step, "is_probation": True 
            }

    # Detailed: Update in-memory scores and statistics for rules used in a batch of examples.
    # Parameters:
    # - usage_data_batch (list[list[(doc, dist, id)]]): For each example, a list of used rule tuples collected during retrieval.
    # - is_correct_list (list[bool]): For each example, whether the model's output after using the rules was correct.
    # - model_outputs (list[str]): The model outputs corresponding to each example, used to fingerprint rule usage.
    # Implementation details: increments `self.current_step`, iterates over used rules, initializes missing stats entries,
    # uses a content fingerprint to filter irrelevant rule matches, adjusts usage counts, history_correct and score.
    # Score increases on correct usages and decays on incorrect usages; penalty magnitude is reduced for long successful history.
    def update_scores_batch(self, usage_data_batch, is_correct_list, model_outputs):
        self.current_step += 1 
        for i, used_rules in enumerate(usage_data_batch):
            if not used_rules: continue
            is_correct = is_correct_list[i]
            output_text = model_outputs[i]
            
            for rule_item in used_rules:
                sid, _, content = rule_item
                if sid not in self.skill_stats:
                    self.skill_stats[sid] = {"score": 0.5, "usage": 0, "history_correct": 0, "created_step": 0, "is_probation": False}
                stats = self.skill_stats[sid]
                
                if len(content) > 10:
                    fingerprint = content[:10] 
                    if fingerprint not in output_text and not is_correct:
                        continue 

                stats['usage'] += 1
                if is_correct:
                    stats['history_correct'] += 1
                    stats['score'] = min(1.0, stats['score'] + 0.1)
                    if stats.get('is_probation') and stats['score'] > 0.6:
                        stats['is_probation'] = False
                else:
                    penalty = 0.2
                    if stats['history_correct'] > 10: penalty = 0.1
                    if stats['history_correct'] > 50: penalty = 0.05
                    stats['score'] = max(0.0, stats['score'] - penalty)

    # Detailed: Prune low-quality rules from memory according to usage and score thresholds.
    # Parameters:
    # - min_usage (int): Minimum usage count before a rule is considered for score-based pruning.
    # - threshold (float): Score threshold below which frequently used rules will be removed.
    # Returns:
    # - Number of deleted rule ids.
    # Implementation details: applies a small decay to scores, removes probationary rules below 0.4,
    # removes rules with usage >= min_usage and score < threshold, and ages out very old unused rules.
    def prune_db(self, min_usage=5, threshold=0.3):
        ids_to_delete = []
        decay_rate = 0.01 
        for sid, stats in list(self.skill_stats.items()):
            if stats['score'] < 0.95:
                stats['score'] -= decay_rate
            if stats.get('is_probation', False):
                if stats['score'] < 0.4:
                    ids_to_delete.append(sid)
                    continue
            if stats['usage'] >= min_usage and stats['score'] < threshold:
                ids_to_delete.append(sid)
                continue
            age = self.current_step - stats.get('created_step', 0)
            if age > 1000 and stats['usage'] < 2:
                ids_to_delete.append(sid)

        if ids_to_delete:
            print(f"✂️ [Prune] Removing {len(ids_to_delete)} low-score rules (remaining: {len(self.skill_stats) - len(ids_to_delete)})")
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.skill_stats[sid]
            return len(ids_to_delete)
        return 0

# ================= 2. vLLM Evolution Trainer (main area to refactor) =================
class ReflexionTrainerFull:
    # Detailed: Initialize the trainer and load runtime components.
    # Parameters: none
    # Implementation details: constructs the vLLM `LLM` instance with model and GPU settings, sets up three
    # sampling parameter sets (`params_inference`, `params_reflection`, `params_verify`), loads a SentenceTransformer
    # embedder on CPU to conserve GPU memory, instantiates `MemoryManager` with reset=True and creates an empty
    # RAG log file at `self.rag_log_path`.
    def __init__(self):
        print("🚀 Initializing training environment (vLLM Accelerated Mode)...")
        
        self.llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            tensor_parallel_size=1,
            max_model_len=2048,
            download_dir=HF_CACHE_DIR
        )
        
        self.params_inference = SamplingParams(
            temperature=0.5, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        self.params_reflection = SamplingParams(
            temperature=0.7, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )
        self.params_verify = SamplingParams(
            temperature=0.1, top_p=0.9, max_tokens=MAX_NEW_TOKENS,
            stop=["<|im_end|>", "<|endoftext|>"]
        )

        print("   -> Loading Embedder (CPU enforced to preserve GPU memory)...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device="cpu", cache_folder=HF_CACHE_DIR)
        
        self.memory = MemoryManager(reset=False)

        self.rag_log_path = "rag_usage_log.jsonl"
        open(self.rag_log_path, "w").close() 

    # Detailed: Batch generation using vLLM.
    # Parameters:
    # - prompts (list[str]): A list of prompt strings formatted for the vLLM `generate` API.
    # - sampling_params: A `SamplingParams` instance controlling temperature, top_p, max_tokens and stop tokens.
    # Returns:
    # - list[str]: Generated output texts (leading/trailing whitespace stripped) for each prompt.
    # Implementation details: calls `self.llm.generate(prompts, sampling_params, use_tqdm=False)` and extracts
    # the first output text for each generation result.
    def batch_generate_vllm(self, prompts, sampling_params):
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    # Detailed: Detect whether strategy text includes problem-specific numeric values copied from the question.
    # Parameters:
    # - text (str): Candidate strategy text returned by the model.
    # - question_text (str): The original question text used to derive a whitelist of numbers to compare against.
    # Returns:
    # - bool: True if any numeric token present in `text` also appears in `question_text` and is not in the allowed whitelist.
    # Implementation details: extracts integer tokens from both strings and compares, excluding common safe values.
    def has_specific_numbers(self, text, question_text):
        q_nums = set(re.findall(r'\b\d+\b', question_text))
        r_nums = set(re.findall(r'\b\d+\b', text))
        whitelist = {'1', '2', '3', '4', '10', '100', '180', '360'} 
        for n in r_nums:
            if n in q_nums and n not in whitelist:
                return True 
        return False

    # Detailed: Clean a reflection text produced by the model.
    # Parameters:
    # - text (str): Raw reflection output from the model which may include markdown fences and conversational phrases.
    # Returns:
    # - str: Cleaned-up text with extraneous patterns removed and duplicate Trigger/Strategy sections truncated.
    # Implementation details: removes markdown fences, strips common starter phrases and trims repeated structured sections.
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

    # Detailed: Parse a cleaned reflection into structured Trigger and Strategy components.
    # Parameters:
    # - text (str): Reflection text returned by the model (will be cleaned internally first).
    # Returns:
    # - tuple(str, str) or None: (trigger_text, strategy_text) if both parts are found and pass simple length checks; otherwise None.
    # Implementation details: searches for labelled sections using regex, normalizes whitespace and truncates overly long triggers.
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
            if len(t_text) > 200: t_text = t_text[:200] 
            if "Strategy" in t_text: t_text = t_text.split("Strategy")[0].strip()
            return t_text, s_text
        return None

    # Detailed: Build a formatted prompt for vLLM, optionally including a strategy context.
    # Parameters:
    # - q (str): The question text to ask the model.
    # - context (str|None): Optional strategy/context string to prepend under [STRATEGY]. If None, a minimal prompt is used.
    # Returns:
    # - str: A prompt string wrapped with the conversation markers expected by the model (<|im_start|>user/...assistant).
    # Implementation details: When context is provided, it is included under [STRATEGY] with strict instructions not to add new examples.
    def construct_prompt(self, q, context=None):
        if context:
            content = f"""
            You may use the following abstract strategy ONLY if it is relevant.
            Do NOT introduce new examples, sub-questions, or conditions.

            [STRATEGY]
            {context}

            [QUESTION]
            {q}

            Answer step-by-step and give ONLY the final answer.
            """
        else:
            content = f"Question: {q}\nAnswer step-by-step:"
        return f"<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"

    # Detailed: Load dataset subset and precompute question embeddings.
    # Parameters:
    # - num_samples (int): Number of samples to select from GSM8K training split for precomputation.
    # Returns:
    # - tuple: (dataset, all_questions_raw, all_answers_raw, all_q_embeddings, total_len)
    # Implementation details: selects the first `num_samples`, prints status, and encodes all questions with the SentenceTransformer
    # (running on CPU) to produce `all_q_embeddings` reused across epochs.
    def prepare_dataset_and_embeddings(self, num_samples=200):
        dataset = load_dataset("gsm8k", "main")['train'].select(range(num_samples))
        total_len = len(dataset)
        print(f"⚡️ Precomputing {total_len} question embeddings (CPU mode)...")
        all_questions_raw = dataset['question']
        all_answers_raw = dataset['answer']
        all_q_embeddings = self.embedder.encode(all_questions_raw, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
        return dataset, all_questions_raw, all_answers_raw, all_q_embeddings, total_len

    # Detailed: Perform zero-shot inference for a chunk of questions and evaluate results.
    # Parameters:
    # - chunk_questions (list[str]): The questions in the current batch/chunk.
    # - chunk_answers (list[str]): Corresponding ground-truth answers for evaluation.
    # Returns:
    # - (zs_is_correct, incorrect_local_indices, zs_outputs):
    #   zs_is_correct (list[bool]) flags for each question indicating correctness under zero-shot.
    #   incorrect_local_indices (list[int]) indices (0-based within the chunk) of incorrect items.
    #   zs_outputs (list[str]) raw generated outputs from the model for further processing if needed.
    # Implementation details: constructs prompts, calls `batch_generate_vllm`, and checks answers using `check_answer`.
    def zero_shot_phase(self, chunk_questions, chunk_answers):
        zero_shot_prompts = [self.construct_prompt(q) for q in chunk_questions]
        zs_outputs = self.batch_generate_vllm(zero_shot_prompts, self.params_inference)

        zs_is_correct = []
        incorrect_local_indices = []
        for idx, pred in enumerate(zs_outputs):
            gt = chunk_answers[idx]
            is_right = self.check_answer(pred, gt)
            zs_is_correct.append(is_right)
            if not is_right:
                incorrect_local_indices.append(idx)
        return zs_is_correct, incorrect_local_indices, zs_outputs

    # Detailed: Perform retrieval-augmented generation (RAG) for items that were incorrect and update memory scores.
    # Parameters:
    # - incorrect_local_indices (list[int]): Indices (0-based within the chunk) of items to attempt RAG on.
    # - chunk_questions (list[str]): The list of questions in the chunk.
    # - chunk_q_embeddings: Array-like embeddings for the chunk questions (used to query memory).
    # - chunk_answers (list[str]): Ground-truth answers for evaluation.
    # - chunk_final_correct (list[bool]): Mutable list marking which items are considered correct; this list may be updated in-place.
    # Returns:
    # - still_incorrect_indices (list[int]): Indices (0-based within the chunk) that remain incorrect after RAG.
    # Implementation details: queries memory for candidate rules, builds prompts for those with rules, runs vLLM on those prompts,
    # evaluates outputs, updates memory scores via `update_scores_batch`, and returns the still-incorrect set.
    def rag_phase(self, incorrect_local_indices, chunk_questions, chunk_q_embeddings, chunk_answers, chunk_final_correct):
        rag_usage_for_update = []
        rag_is_correct_for_update = []
        rag_outputs_for_update = []

        still_incorrect_indices = []

        if not incorrect_local_indices:
            return still_incorrect_indices

        wrong_questions = [chunk_questions[i] for i in incorrect_local_indices]
        wrong_embeddings = chunk_q_embeddings[incorrect_local_indices]
        wrong_answers = [chunk_answers[i] for i in incorrect_local_indices]

        retrieved_batch = self.memory.batch_retrieve(wrong_embeddings, top_k=3, threshold=0.4)

        index_to_rules = {}
        rag_prompts = []
        for k, q in enumerate(wrong_questions):
            rules_list = retrieved_batch[k]
            original_idx = incorrect_local_indices[k]
            
            if rules_list:
                print(f"\n📚 [RAG] Retrieved Principles for Question: {q[:50]}...")
                for r in rules_list:
                    print(f"   - {r[0][:100]}...")
                context_text = "\n".join([f"- {r[0]}" for r in rules_list])
                rag_prompts.append(self.construct_prompt(q, context_text))
                rag_usage_for_update.append(rules_list)
                index_to_rules[original_idx] = [r[0] for r in rules_list]
            else:
                rag_usage_for_update.append([])
                rag_prompts.append(None)
                still_incorrect_indices.append(incorrect_local_indices[k])
                index_to_rules[original_idx] = []

        real_rag_prompts = [p for p in rag_prompts if p is not None]
        if real_rag_prompts:
            real_rag_outputs = self.batch_generate_vllm(real_rag_prompts, self.params_inference)

        output_cursor = 0
        for k, q in enumerate(wrong_questions):
            if rag_prompts[k] is None:
                continue

            pred = real_rag_outputs[output_cursor]
            output_cursor += 1

            gt = wrong_answers[k]
            is_right = self.check_answer(pred, gt)

            rag_is_correct_for_update.append(is_right)
            rag_outputs_for_update.append(pred)

            if is_right:
                chunk_final_correct[incorrect_local_indices[k]] = True
            else:
                still_incorrect_indices.append(incorrect_local_indices[k])

        final_usage = [rag_usage_for_update[k] for k in range(len(wrong_questions)) if rag_prompts[k] is not None]
        if final_usage:
            self.memory.update_scores_batch(final_usage, rag_is_correct_for_update, rag_outputs_for_update)

        return still_incorrect_indices, index_to_rules

    # Detailed: Generate reflection-based rules for still-incorrect items and verify them before adding to memory.
    # Parameters:
    # - still_incorrect_indices (list[int]): Indices (0-based within the chunk) of items that remain incorrect after RAG.
    # - chunk_questions (list[str]): Questions in the chunk.
    # - chunk_answers (list[str]): Ground-truth answers for verification.
    # - chunk_final_correct (list[bool]): Mutable list to be updated in-place when reflections verify successfully.
    # Implementation details: Builds strict-format reflection prompts, parses candidate triggers/strategies, filters
    # candidates that leak specific numbers, verifies possible rules by applying them via the model, and stores verified
    # rules into memory with embeddings.
    def reflection_phase(self, still_incorrect_indices, chunk_questions, chunk_answers, chunk_final_correct):
        if not still_incorrect_indices:
            return

        reflect_prompts = []
        verify_data = []
        for idx in still_incorrect_indices:
            q = chunk_questions[idx]
            gt = chunk_answers[idx]
            p_content = f"""Task: Extract a general math rule from the incorrect problem.
            STRICT FORMAT INSTRUCTION:
            1. Do NOT print "Sure", "Here is", or any conversational filler.
            2. Do NOT use markdown code blocks.
            3. Start directly with "**Trigger (A)**".
            4. Use variables like X, Y, Z instead of specific numbers.
            5. STOP after providing the Strategy.
            [EXAMPLE]
            Problem: John buys 5 apples for $2 each. Total cost?
            Solution: 5 * 2 = 10.
            **Trigger (A)**: Calculating total cost from quantity and rate.
            **Strategy (B)**: Total Cost = Quantity * Unit Price.
            [YOUR TURN]
            Problem: {q}
            Solution: {gt}
            Summarize the rule:"""
            reflect_prompts.append(f"<|im_start|>user\n{p_content}<|im_end|>\n<|im_start|>assistant\n")
            verify_data.append((q, gt))

        reflections = self.batch_generate_vllm(reflect_prompts, self.params_reflection)

        temp_candidates = []
        for k, text in enumerate(reflections):
            parsed = self.parse_reflection(text)
            if parsed:
                p_text, s_text = parsed
                orig_q = verify_data[k][0]
                if self.has_specific_numbers(s_text, orig_q):
                    continue
                temp_candidates.append((p_text, s_text, k))

        if not temp_candidates:
            return

        verify_prompts = []
        for p_text, s_text, k in temp_candidates:
            orig_q = verify_data[k][0]
            vp_content = f"Rule: {s_text}\nQuestion: {orig_q}\nAnswer step-by-step:"
            verify_prompts.append(f"<|im_start|>user\n{vp_content}<|im_end|>\n<|im_start|>assistant\n")

        verify_outputs = self.batch_generate_vllm(verify_prompts, self.params_verify)

        verified_patterns = []
        verified_strategies = []
        for m, pred in enumerate(verify_outputs):
            k_idx = temp_candidates[m][2]
            orig_gt = verify_data[temp_candidates[m][2]][1]
            if self.check_answer(pred, orig_gt):
                verified_patterns.append(temp_candidates[m][0])
                verified_strategies.append(temp_candidates[m][1])
                original_global_idx = still_incorrect_indices[k_idx]
                chunk_final_correct[original_global_idx] = True

        if verified_patterns:
            p_embeds = self.embedder.encode(verified_patterns, convert_to_numpy=True).tolist()
            self.memory.add_experience_batch(verified_patterns, verified_strategies, p_embeds)
    
    def self_explore_phase(self, still_incorrect_indices, chunk_questions, chunk_answers, index_to_rules=None):
        """
        使用 Self-Explore 机制：
        1. 重试错误的题目
        2. 如果依然错误，按步骤验证 (Verify Step)
        3. 找到第一个错误步骤，生成 Principle
        4. 将 Principle 解析为 Trigger/Strategy 存入 Memory
        """
        if not still_incorrect_indices:
            return

        print(f"   🔍 Self-Exploring {len(still_incorrect_indices)} failed samples...")

        # 1. 准备数据
        target_questions = [chunk_questions[i] for i in still_incorrect_indices]
        target_answers = [chunk_answers[i] for i in still_incorrect_indices]
        
        # 2. 重新生成一次完整的解题过程 (Trace) 用于分析
        # 我们使用 params_inference (temp=0.5) 或者稍微高一点的 temp
        prompts = [self.construct_prompt(q) for q in target_questions]
        traces = self.batch_generate_vllm(prompts, self.params_inference)

        new_patterns = []
        new_strategies = []
        new_embeddings_inputs = [] # 用于生成 embedding 的文本 (通常是 Trigger)

        for i, trace in enumerate(traces):
            gt = target_answers[i]
            q = target_questions[i]
            original_idx = still_incorrect_indices[i]
            
            # 如果这次蒙对了，就跳过分析
            if self.check_answer(trace, gt):
                continue

            # 3. 分割步骤 (简单的按换行符分割)
            steps = [s.strip() for s in trace.split('\n') if s.strip()]
            if not steps: continue

            # 4. 逐步验证
            # 基础 Prompt
            current_prompt = self.construct_prompt(q).split("<|im_start|>assistant\n")[0] + "<|im_start|>assistant\n"
            
            found_error = False
            error_step_content = ""
            partial_reasoning = ""

            # 限制验证步数，防止太慢
            for step_idx, step in enumerate(steps):
                # 构建包含当前步骤的 Prompt
                verify_prompt = current_prompt + step
                
                # 验证这一步是否走偏了
                # 注意：如果这是最后一步，verify_step 实际上就是在验证最终答案
                is_valid = self.verify_step_vllm(verify_prompt, gt, k=5)
                
                if not is_valid:
                    # 找到错误步骤！
                    found_error = True
                    error_step_content = step
                    partial_reasoning = "\n".join(steps[:step_idx]) if step_idx > 0 else "(No previous steps)"
                    print(f"      Found error at step {step_idx+1}")
                    break
                
                # 如果正确，更新 prompt 继续验证下一步
                current_prompt += step + "\n"

            # 5. 如果找到了错误步骤，生成 Principle
            if found_error:
                MAX_PRINCIPLE_RETRIES = 3
                for attempt in range(MAX_PRINCIPLE_RETRIES):
                    principle_text = self.generate_principle_vllm(q, partial_reasoning, error_step_content, gt)

                    print(f"\n{'='*40}")
                    print(f"📝 [Self-Explore Log] (Attempt {attempt+1}/{MAX_PRINCIPLE_RETRIES})")
                    print(f"❓ Question: {q}")
                    
                    # Print used RAG principles
                    if index_to_rules and original_idx in index_to_rules and index_to_rules[original_idx]:
                        print(f"📚 RAG Principles Used:")
                        for r in index_to_rules[original_idx]:
                            print(f"   - {r[:100]}...")
                    else:
                        print(f"📚 RAG Principles Used: None")

                    print(f"✅ Ground Truth: {gt}")
                    print(f"🤖 Model Output (Trace):\n{trace}")
                    print(f"❌ Error Step: {error_step_content}")
                    print(f"💡 Generated Principle (Raw):\n{principle_text}")
                    print(f"{'='*40}\n")
                    
                    # 6. 解析 Principle (格式: "To solve [Trigger], consider [Strategy].")
                    match = re.search(r"To solve\s+(.*?),\s*consider\s+(.*)", principle_text, re.IGNORECASE | re.DOTALL)
                    
                    trigger = ""
                    strategy = ""
                    
                    if match:
                        trigger = match.group(1).strip().rstrip('.')
                        strategy = match.group(2).strip().rstrip('.')
                    else:
                        strategy = principle_text
                        trigger = "Solving this type of problem"

                    # 简单的过滤
                    if len(strategy) > 10 and not self.has_specific_numbers(strategy, q):
                        # 7. 验证 Principle
                        print(f"      🕵️ Verifying Principle...")
                        verify_prompt = self.construct_prompt(q, context=strategy)
                        verify_output = self.batch_generate_vllm([verify_prompt], self.params_inference)[0]
                        
                        if self.check_answer(verify_output, gt):
                            print(f"      ✅ Verification Passed! Adding rule.")
                            print(f"      💡 Generated Rule: {trigger} -> {strategy[:50]}...")
                            new_patterns.append(trigger)
                            new_strategies.append(strategy)
                            new_embeddings_inputs.append(trigger) 
                            break 
                        else:
                            print(f"      ❌ Verification Failed. Model output: {verify_output[:50]}...")
                    else:
                        print(f"      ⚠️ Invalid Principle Format/Content.")

        # 7. 批量添加到内存
        if new_patterns:
            print(f"      💾 Adding {len(new_patterns)} new rules to memory...")
            # 生成 Embeddings
            embeddings = self.embedder.encode(new_embeddings_inputs, convert_to_numpy=True)
            self.memory.add_experience_batch(new_patterns, new_strategies, embeddings)


    # Detailed: Run periodic pruning of the memory DB according to the current chunk position.
    # Parameters:
    # - chunk_start (int): The starting index of the current chunk within the epoch; used to compute a pruning cadence.
    # Implementation details: triggers `self.memory.prune_db(threshold=0.25)` every 5 chunks.
    def periodic_prune(self, chunk_start):
        if (chunk_start // CHUNK_SIZE) % 5 == 0:
            self.memory.prune_db(threshold=0.25)

    
    def verify_step_vllm(self, prompt, ground_truth, k=5):
    # 采样 k 次，看是否有一次能做对
    # 注意：这里使用较高的 temperature 来增加探索性
        params = SamplingParams(n=k, temperature=0.7, max_tokens=MAX_NEW_TOKENS)
        
        # vLLM 接受 list[str]
        outputs = self.llm.generate([prompt], params, use_tqdm=False)
        
        # 检查生成的 k 个结果中是否有正确的
        for output in outputs[0].outputs:
            pred_text = output.text
            if self.check_answer(pred_text, ground_truth):
                return True
        return False
    
    def generate_principle_vllm(self, question, partial_reasoning, error_step, answer):
        prompt_content = GENERATE_PRINCIPLE_PROMPT.format(
            question=question,
            partial_reasoning=partial_reasoning,
            error_step=error_step,
            answer=answer
        )
        
        # 构建对话格式
        prompt = f"<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n<|im_start|>user\n{prompt_content}<|im_end|>\n<|im_start|>assistant\n"
        
        # 生成原则
        outputs = self.batch_generate_vllm([prompt], self.params_reflection)
        return outputs[0]
    
    # Detailed: Execute the full evolution training loop across multiple epochs.
    # Parameters: none
    # Implementation details: loads data and embeddings, iterates epochs and chunks, runs zero-shot, RAG and reflection phases,
    # periodically prunes the memory DB, calculates epoch accuracy, updates `best_acc` and enforces `MAX_EPOCHS` and `TARGET_ACCURACY` stopping conditions.
    def run_full_evolution(self):
        dataset, all_questions_raw, all_answers_raw, all_q_embeddings, total_len = self.prepare_dataset_and_embeddings(num_samples=200)

        print(f"🔥 Starting full evolution training (vLLM Speedup) - Target Accuracy: {TARGET_ACCURACY}%")

        epoch = 0
        best_acc = 0.0

        while best_acc < TARGET_ACCURACY and epoch < MAX_EPOCHS:
            epoch += 1
            print(f"\n======== Epoch {epoch}/{MAX_EPOCHS} ========")

            indices = np.random.permutation(total_len)

            epoch_correct_count = 0
            epoch_total_count = 0

            pbar = tqdm(range(0, total_len, CHUNK_SIZE), desc=f"Epoch {epoch} Training")

            for chunk_start in pbar:
                chunk_end = min(chunk_start + CHUNK_SIZE, total_len)
                current_batch_indices = indices[chunk_start:chunk_end]

                chunk_questions = [all_questions_raw[i] for i in current_batch_indices]
                chunk_answers = [all_answers_raw[i] for i in current_batch_indices]
                chunk_q_embeddings = all_q_embeddings[current_batch_indices]
                # Phase 1: Zero-shot inference
                zs_is_correct, incorrect_local_indices, zs_outputs = self.zero_shot_phase(chunk_questions, chunk_answers)
                epoch_total_count += len(chunk_questions)

                chunk_final_correct = zs_is_correct[:]

                # Phase 2: RAG recomputation for incorrect items
                still_incorrect_indices, index_to_rules = self.rag_phase(incorrect_local_indices, chunk_questions, chunk_q_embeddings, chunk_answers, chunk_final_correct)

                # Phase 3: Reflection (for items still incorrect after RAG)
                if still_incorrect_indices:
                     self.self_explore_phase(still_incorrect_indices, chunk_questions, chunk_answers, index_to_rules)

                # Periodic pruning
                self.periodic_prune(chunk_start)

                epoch_correct_count += sum(chunk_final_correct)

                batch_acc = sum(chunk_final_correct) / len(chunk_questions) * 100
                pbar.set_postfix({"Total Acc": f"{batch_acc:.1f}%", "DB": self.memory.collection.count()})

            current_epoch_acc = (epoch_correct_count / epoch_total_count) * 100
            print(f"\n📊 Epoch {epoch} complete | Overall Accuracy (ZS+RAG+Reflect): {current_epoch_acc:.2f}% (Target: {TARGET_ACCURACY}%)")

            if current_epoch_acc > best_acc:
                best_acc = current_epoch_acc

            if current_epoch_acc >= TARGET_ACCURACY:
                print(f"🎉 Congrats! Target accuracy reached ({current_epoch_acc:.2f}%), training finished!")
                break
            elif epoch >= MAX_EPOCHS:
                print("⚠️ Maximum epoch limit reached, stopping training.")
                break
            else:
                print("🔄 Performance below target, continuing to next epoch...")

    # Detailed: Extract the last numeric token from a text string and convert to float.
    # Parameters:
    # - text (str): Input text which may contain numbers, possibly with commas.
    # Returns:
    # - float or None: The numeric value of the last matched token, or None if no numeric tokens found.
    # Implementation details: removes commas and matches integers or decimals using regex, returning the last match as float.
    def extract_number(self, text):
        if not text: return None
        text = text.replace(',', '')
        matches = re.findall(r'-?\d+\.?\d*', text)
        if matches: return float(matches[-1])
        return None

    # Detailed: Compare numeric answers extracted from prediction and ground truth with high precision.
    # Parameters:
    # - pred (str): Model's predicted answer text.
    # - gt (str): Ground truth answer text (may contain '####' marker). If '####' present, the numeric part after it is used.
    # Returns:
    # - bool: True if both numeric values are present and their absolute difference is less than 1e-4.
    # Implementation details: uses `extract_number` to parse floats from both strings and performs absolute difference check.
    def check_answer(self, pred, gt):
        if "####" in gt:
            gold = self.extract_number(gt.split("####")[1])
        else:
            gold = self.extract_number(gt)
        pred_num = self.extract_number(pred)
        if gold is None or pred_num is None: return False
        return abs(gold - pred_num) < 1e-4

    # Detailed: Cleanup heavy runtime objects and free GPU memory.
    # Parameters: none
    # Implementation details: deletes `self.llm` reference if present, runs garbage collection, empties CUDA cache,
    # and attempts to shutdown Ray if it was initialized. Prints progress messages to the console.
    def cleanup(self):
        print("🧹 Cleaning up GPU memory and vLLM processes...")
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
        print("✅ Cleanup complete!")

if __name__ == "__main__":
    trainer = None
    try:
        trainer = ReflexionTrainerFull()
        trainer.run_full_evolution()
    except KeyboardInterrupt:
        print("\n🛑 User interrupted training (KeyboardInterrupt)")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        raise e
    finally:
        if trainer is not None:
            trainer.cleanup()