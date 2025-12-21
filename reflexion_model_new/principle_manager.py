import os
import shutil
import time
import random
import json
import numpy as np
import chromadb
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Any, Union

# ==============================================================================
# SYSTEM CONSTANTS (Physical Constraints, not Tuning Knobs)
# ==============================================================================
DB_PATH = "./reflexion_full_db"
MEMORY_CAPACITY = 2000          # Maximum number of rules the system can hold
DUPLICATE_DIST_LIMIT = 0.15     # Vector distance to consider two rules identical

class MemoryManager:
    """
    An elegant, robust Memory Manager utilizing Bayesian Statistics (Thompson Sampling).
    
    Principles:
    1.  **Parameter-Free Learning**: Instead of arbitrary scores (0.1, 0.5), it uses 
        Beta Distributions (Alpha/Beta) to model the probability of a rule being correct.
    2.  **Intrinsic Exploration**: New rules have high variance (uncertainty), allowing 
        them to naturally rise to the top for testing (Thompson Sampling).
    3.  **Automatic Pruning**: Uses a fixed capacity model. When full, the system 
        evicts rules with the lowest Expected Value (Probability of Success).
    4.  **Robustness**: Handles input edge cases and persists state automatically.
    """

    # Purpose: Initialize the manager, load persistence, and restore Bayesian priors.
    # Arguments: 
    #   - reset: If True, deletes the existing database to start fresh.
    # Side effects: Loads stats into self.stats.
    def __init__(self, reset: bool = False):
        if reset and os.path.exists(DB_PATH):
            try:
                shutil.rmtree(DB_PATH)
                print(f"[Init] Database reset at {DB_PATH}")
            except OSError as e:
                print(f"[Init Warning] Could not delete DB: {e}")

        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        # In-memory stats: {rule_id: {'alpha': float, 'beta': float, 'usage': int}}
        # Alpha = Successes + 1, Beta = Failures + 1
        self.stats = {} 
        self.current_step = 0
        
        self._restore_state()

    # Purpose: internal helper to restore Alpha/Beta priors from disk to memory.
    def _restore_state(self):
        try:
            data = self.collection.get(include=["metadatas"])
            if not data or not data['ids']: return

            for sid, meta in zip(data['ids'], data['metadatas']):
                if meta:
                    self.stats[sid] = {
                        "alpha": float(meta.get("alpha", 1.0)),
                        "beta": float(meta.get("beta", 1.0)),
                        "usage": int(meta.get("usage", 0)),
                        "created_step": int(meta.get("created_step", 0))
                    }
                    if self.stats[sid]["created_step"] > self.current_step:
                        self.current_step = self.stats[sid]["created_step"]
            
            print(f"[Init] Restored Bayesian priors for {len(self.stats)} rules.")
        except Exception as e:
            print(f"[Init Warning] State restoration failed: {e}")

    # Purpose: Retrieve rules using Thompson Sampling (Similarity * Sampled_Quality).
    # Arguments:
    #   - query_embeddings: List of vectors (or single vector).
    #   - top_k: Number of results to return.
    #   - threshold: (Optional) Min similarity floor.
    # Returns: List[List[Tuple(document, distance, id)]]
    def batch_retrieve(self, query_embeddings, top_k: int = 3, threshold: float = 0.5) -> List[List[Tuple[str, float, str]]]:
        # Robust Input Handling: Ensure input is a list of embeddings
        if isinstance(query_embeddings, np.ndarray):
            query_embeddings = query_embeddings.tolist()

        if not isinstance(query_embeddings, list): 
            query_embeddings = [query_embeddings]
        elif query_embeddings and isinstance(query_embeddings[0], (int, float, np.floating)): 
            query_embeddings = [query_embeddings]

        count = self.collection.count()
        if count == 0: return [[] for _ in query_embeddings]

        # 1. Broad Fetch: Get enough candidates to perform re-ranking
        try:
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=min(count, top_k * 5),
                include=['documents', 'distances', 'embeddings']
            )
        except Exception as e:
            print(f"[Retrieve Error] Chroma query failed: {e}")
            return [[] for _ in query_embeddings]

        final_results = []

        # 2. Bayesian Reranking using Thompson Sampling
        # Zip ensures we don't crash if Chroma returns inconsistent list lengths
        for ids, docs, dists in zip(results.get('ids', []), results.get('documents', []), results.get('distances', [])):
            ranked_candidates = []
            
            for i in range(len(ids)):
                sid = ids[i]
                dist = dists[i]
                
                # Convert L2 Distance to Similarity (approx 0 to 1)
                similarity = 1.0 / (1.0 + dist)
                
                # Get Bayesian Priors (Default to Uniform 1.0/1.0 if missing)
                stat = self.stats.get(sid, {'alpha': 1.0, 'beta': 1.0})
                
                # Thompson Sampling: Sample from Beta Distribution
                # High variance (new rules) -> chance to sample high
                quality_sample = np.random.beta(stat['alpha'], stat['beta'])
                
                # Combined Score
                score = similarity * quality_sample
                
                # Soft threshold: Skip if extremely distant, unless it's a "super rule" (EV > 0.9)
                expected_val = stat['alpha'] / (stat['alpha'] + stat['beta'])
                min_sim = 1.0 / (1.0 + threshold)
                if similarity < min_sim and expected_val < 0.9:
                    continue

                ranked_candidates.append((-score, docs[i], dist, sid)) # Negate score for min-heap style sort
            
            # Sort by Score Descending
            ranked_candidates.sort(key=lambda x: x[0]) 
            
            # Extract top_k
            final_results.append([(item[1], item[2], item[3]) for item in ranked_candidates[:top_k]])

        # Fill empty slots if any mismatch occurred
        while len(final_results) < len(query_embeddings):
            final_results.append([])

        return final_results

    # Purpose: Add new rules. Manages Capacity (Eviction) and Duplicates.
    # Arguments:
    #   - patterns: List of rule patterns strings.
    #   - strategies: List of strategy strings.
    #   - embeddings: List of vectors.
    #   - source_q_hashes: List of hashes identifying the source question.
    def add_experience_batch(self, patterns: List[str], strategies: List[str], embeddings, source_q_hashes: List[str] = None):
        if not patterns: return
        
        # 1. Capacity Management (Garbage Collection)
        # If adding these rules exceeds capacity, remove the statistically "worst" rules first
        if len(self.stats) + len(patterns) > MEMORY_CAPACITY:
            needed = (len(self.stats) + len(patterns)) - MEMORY_CAPACITY
            # Sort by Expected Value: Alpha / (Alpha + Beta)
            sorted_stats = sorted(
                self.stats.items(), 
                key=lambda item: item[1]['alpha'] / (item[1]['alpha'] + item[1]['beta'])
            )
            ids_to_remove = [item[0] for item in sorted_stats[:needed]]
            
            if ids_to_remove:
                self.collection.delete(ids=ids_to_remove)
                for sid in ids_to_remove: del self.stats[sid]

        # 2. Add New Rules
        ids_add, embs_add, docs_add, metas_add = [], [], [], []

        for i, emb in enumerate(embeddings):
            # Duplicate Check: Don't add if semantically identical rule exists
            try:
                exist = self.collection.query(query_embeddings=[emb], n_results=1)
                if exist['ids'] and exist['ids'][0]:
                    if exist['distances'][0][0] < DUPLICATE_DIST_LIMIT:
                        # Reinforce the existing rule instead
                        sid = exist['ids'][0][0]
                        if sid in self.stats:
                            self.stats[sid]['alpha'] += 0.5 # Slight boost for rediscovery
                        continue
            except: pass

            new_id = f"rule_{int(time.time())}_{i}_{random.randint(0,999)}"
            # Initial Prior: Uniform Distribution (Alpha=1, Beta=1) -> "I know nothing"
            meta = {
                "alpha": 1.0, 
                "beta": 1.0, 
                "usage": 0, 
                "created_step": self.current_step,
                "pattern": patterns[i],
                "status": "probation", # Default to probation
                "source_q_hash": source_q_hashes[i] if source_q_hashes else ""
            }
            
            ids_add.append(new_id)
            embs_add.append(emb)
            docs_add.append(strategies[i])
            metas_add.append(meta)
            self.stats[new_id] = meta.copy()

        if ids_add:
            self.collection.add(ids=ids_add, embeddings=embs_add, documents=docs_add, metadatas=metas_add)

    # Purpose: Update Bayesian Priors based on outcome.
    # Arguments:
    #   - usage_data: List of (doc, dist, id) tuples.
    #   - is_correct_list: Boolean outcomes.
    #   - model_outputs: Text for fuzzy verification.
    #   - current_q_hashes: List of hashes for the questions being solved.
    def update_scores_batch(self, usage_data, is_correct_list, model_outputs, current_q_hashes: List[str] = None):
        self.current_step += 1
        ids_sync, metas_sync = [], []

        # Zip safely to avoid index errors
        for i, (used_rules, is_correct, output_txt) in enumerate(zip(usage_data, is_correct_list, model_outputs)):
            if not used_rules: continue
            
            current_q_hash = current_q_hashes[i] if current_q_hashes else ""
            
            for rule_item in used_rules:
                # Unpack safely (expecting doc, dist, id OR id, dist, content depending on caller)
                # Assuming structure: (doc_content, distance, rule_id)
                if len(rule_item) == 3:
                    content, _, sid = rule_item
                else: continue

                if sid not in self.stats: continue
                
                # 1. Fuzzy Verification: Did the model actually use this rule?
                # If text overlap is too low (< 40%) AND result is wrong, don't penalize.
                try:
                    match = SequenceMatcher(None, content, output_txt).find_longest_match(0, len(content), 0, len(output_txt))
                    overlap_ratio = match.size / len(content)
                    if overlap_ratio < 0.4 and not is_correct:
                        continue
                except: pass

                # 2. Bayesian Update
                self.stats[sid]['usage'] += 1
                if is_correct:
                    self.stats[sid]['alpha'] += 1.0
                    
                    # Graduation Logic: If correct AND different question, graduate from probation
                    if self.stats[sid].get("status", "active") == "probation":
                        src_hash = self.stats[sid].get("source_q_hash", "")
                        if current_q_hash and src_hash and current_q_hash != src_hash:
                            self.stats[sid]["status"] = "active"
                            # print(f"🎉 Rule {sid} graduated from probation!")
                else:
                    self.stats[sid]['beta'] += 1.0

                ids_sync.append(sid)
                metas_sync.append({
                    "alpha": self.stats[sid]['alpha'],
                    "beta": self.stats[sid]['beta'],
                    "usage": self.stats[sid]['usage'],
                    "created_step": self.stats[sid]['created_step'],
                    "pattern": self.stats[sid].get("pattern", ""),
                    "status": self.stats[sid].get("status", "active"),
                    "source_q_hash": self.stats[sid].get("source_q_hash", "")
                })

        # 3. Persist State
        if ids_sync:
            try:
                self.collection.update(ids=ids_sync, metadatas=metas_sync)
            except Exception as e:
                print(f"[Update Error] {e}")

    def prune_probationary_rules(self):
        """
        Remove all rules that are still in 'probation' status.
        Should be called at the end of training.
        """
        ids_to_delete = []
        for sid, stat in list(self.stats.items()):
            if stat.get("status") == "probation":
                ids_to_delete.append(sid)
        
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.stats[sid]
            print(f"[Prune] Removed {len(ids_to_delete)} probationary rules (failed to graduate).")
        else:
            print("[Prune] No probationary rules found to remove.")

    # Purpose: Manual cleanup (Optional, as capacity handles most pruning).
    # Allows removing specifically bad performers below a certain Expected Value.
    # Returns: Number of deleted items.
    def prune_db(self, min_usage: int = 5, threshold: float = 0.3) -> int:
        ids_to_delete = []
        
        for sid, stat in list(self.stats.items()):
            # Calculate Expected Value (Mean of Beta Distribution)
            ev = stat['alpha'] / (stat['alpha'] + stat['beta'])
            
            # Prune if significant data exists (usage > min) AND EV is low
            if stat['usage'] >= min_usage and ev < threshold:
                ids_to_delete.append(sid)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.stats[sid]
            print(f"[Prune] Removed {len(ids_to_delete)} low-value rules.")
            return len(ids_to_delete)
        return 0

    # Purpose: Debug utility to inspect memory state.
# Purpose: Export a highly detailed snapshot of the memory state for debugging.
    # Fetches text content from DB and merges with in-memory Bayesian stats.
    def dump_memory_snapshot(self, filepath="memory_debug.json"):
        # 1. Fetch text content (Documents) from ChromaDB
        try:
            # We fetch all data to ensure we match IDs to actual text strategies
            db_data = self.collection.get(include=['documents', 'metadatas'])
            id_to_doc = dict(zip(db_data['ids'], db_data['documents']))
            
            # Also fetch patterns from DB metadata if missing in memory (redundancy check)
            id_to_meta_pattern = {
                sid: meta.get('pattern', "N/A") 
                for sid, meta in zip(db_data['ids'], db_data['metadatas'])
            }
        except Exception as e:
            print(f"[Dump Error] Could not fetch data from DB: {e}")
            id_to_doc = {}
            id_to_meta_pattern = {}

        # 2. Build Detailed Rule List
        detailed_rules = []
        for sid, stat in self.stats.items():
            # Basic Stats
            alpha = stat['alpha']
            beta = stat['beta']
            total = alpha + beta
            
            # Bayesian Metrics
            # Mean (Expected Value): The probability of being correct [0, 1]
            expected_value = alpha / total
            
            # Variance: Measure of Uncertainty. 
            # High Variance = Low Confidence (Need more exploration).
            # Low Variance = High Confidence (Established rule).
            # Var = (α * β) / ((α + β)^2 * (α + β + 1))
            variance = (alpha * beta) / ((total ** 2) * (total + 1))
            
            # Derived Status Tag
            if stat['usage'] < 3:
                status = "NEW / EXPLORING"
            elif expected_value > 0.8:
                status = "PROVEN / HIGH QUALITY"
            elif expected_value < 0.3:
                status = "FAILED / LOW QUALITY"
            elif variance > 0.05:
                status = "UNCERTAIN / VOLATILE"
            else:
                status = "AVERAGE"

            # Resolve Pattern (Memory > DB Metadata > Fallback)
            pattern_text = stat.get("pattern") or id_to_meta_pattern.get(sid, "N/A")

            detailed_rules.append({
                "rank_metrics": {
                    "expected_value": round(expected_value, 4),
                    "uncertainty_variance": round(variance, 4),
                    "status": status
                },
                "content": {
                    "pattern_trigger": pattern_text,
                    "strategy_solution": id_to_doc.get(sid, "[MISSING DOC]"),
                    "rule_id": sid
                },
                "raw_stats": {
                    "alpha_success": alpha,
                    "beta_failure": beta,
                    "usage_count": stat['usage'],
                    "raw_win_rate": f"{((alpha-1) / max(1, stat['usage'])) * 100:.1f}%" if stat['usage'] > 0 else "N/A"
                },
                "lifecycle": {
                    "created_at_step": stat['created_step'],
                    "current_age": self.current_step - stat['created_step']
                }
            })

        # 3. Sort by Expected Value (Best to Worst)
        detailed_rules.sort(key=lambda x: x['rank_metrics']['expected_value'], reverse=True)

        # 4. Add Rank Index
        for idx, rule in enumerate(detailed_rules):
            rule['rank_index'] = idx + 1

        # 5. Construct Final JSON Structure
        snapshot = {
            "system_summary": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_steps": self.current_step,
                "total_rules": len(detailed_rules),
                "memory_capacity": MEMORY_CAPACITY,
                "capacity_usage": f"{(len(detailed_rules)/MEMORY_CAPACITY)*100:.1f}%"
            },
            "rules": detailed_rules
        }

        # 6. Write to File
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2, ensure_ascii=False)
            print(f"[Debug] Detailed memory snapshot saved to {filepath}")
        except Exception as e:
            print(f"[Dump Error] Failed to write file: {e}")