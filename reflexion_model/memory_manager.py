from config import * 
import os
import shutil
import time
import random
import chromadb

class MemoryManager:
    def __init__(self, reset=False):
        if reset and os.path.exists(DB_PATH):
            shutil.rmtree(DB_PATH)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(name="rule_book")
        
        # 内存缓存 Stats
        self.skill_stats = {} 
        self.current_step = 0 # 全局计数器

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
            print(f"✂️ [淘汰] 清理 {len(ids_to_delete)} 条低分规则 (剩余: {len(self.skill_stats) - len(ids_to_delete)})")
            self.collection.delete(ids=ids_to_delete)
            for sid in ids_to_delete:
                del self.skill_stats[sid]
            return len(ids_to_delete)
        return 0