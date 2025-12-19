import os
import shutil
import time
import random
import chromadb
from collections import Counter

class PrincipleManager:
    """
    PrincipleManager is a unified class for managing principles (rules and strategies) stored in a ChromaDB database.
    It provides functionality for adding, retrieving, updating, and pruning principles, ensuring efficient and relevant storage.
    """

    def __init__(self, db_path="./reflexion_full_db", reset=False):
        """
        Initialize the PrincipleManager with optional database reset.

        Parameters:
        - db_path (str): Path to the database directory.
        - reset (bool): If True, clears the existing database at db_path before initializing.
        """
        self.db_path = db_path
        if reset and os.path.exists(self.db_path):
            shutil.rmtree(self.db_path)
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(name="principle_book")
        self.principle_stats = {}
        self.current_step = 0

    def _query_embeddings(self, embeddings, top_k):
        """
        Internal helper to query the database for similar embeddings.

        Parameters:
        - embeddings (list): List of embeddings to query.
        - top_k (int): Number of top results to return.

        Returns:
        - list: A list of lists containing tuples (doc, dist, id) for each query.
        """
        count = self.collection.count()
        if count == 0:
            return [[] for _ in embeddings]
        try:
            results = self.collection.query(query_embeddings=embeddings, n_results=min(top_k, count))
            return [
                [(results['documents'][i][j], results['distances'][i][j], results['ids'][i][j])
                 for j in range(len(results['ids'][i]))]
                for i in range(len(embeddings))
            ]
        except Exception:
            return [[] for _ in embeddings]

    def retrieve_principles(self, embeddings, top_k=3):
        """
        Retrieve the most similar principles for a batch of query embeddings.

        Parameters:
        - embeddings (list): A list of embeddings representing the queries.
        - top_k (int): The number of top results to return for each query.

        Returns:
        - list: A list of lists containing tuples (doc, dist, id) for each query.
        """
        return self._query_embeddings(embeddings, top_k)

    def add_principles(self, patterns, strategies, embeddings):
        """
        Add a batch of new principles (patterns and strategies) to the database.

        Parameters:
        - patterns (list): A list of trigger patterns (e.g., problem-solving heuristics).
        - strategies (list): A list of corresponding strategies (e.g., step-by-step solutions).
        - embeddings (list): A list of embeddings representing the patterns.
        """
        unique_patterns, unique_strategies, unique_embeddings = [], [], []

        for i, emb in enumerate(embeddings):
            existing = self._query_embeddings([emb], top_k=1)[0]
            if existing and existing[0][1] < 0.15:  # Check if similar principle exists
                principle_id = existing[0][2]
                if principle_id in self.principle_stats:
                    self.principle_stats[principle_id]['score'] = min(1.0, self.principle_stats[principle_id]['score'] + 0.05)
                continue
            unique_patterns.append(patterns[i])
            unique_strategies.append(strategies[i])
            unique_embeddings.append(emb)

        if not unique_patterns:
            return

        new_ids = [f"principle_{int(time.time())}_{i}_{random.randint(0, 999)}" for i in range(len(unique_patterns))]
        self.collection.add(
            ids=new_ids,
            embeddings=unique_embeddings,
            documents=unique_strategies,
            metadatas=[{"pattern": p} for p in unique_patterns]
        )

        for principle_id in new_ids:
            self.principle_stats[principle_id] = {
                "score": 0.5,
                "usage": 0,
                "history_correct": 0,
                "created_step": self.current_step,
                "is_probation": True
            }

    def update_principle_scores(self, usage_data, correctness_list):
        """
        Update the scores and statistics for a batch of used principles.

        Parameters:
        - usage_data (list): A list of tuples (id, dist, content) representing the used principles.
        - correctness_list (list): A list of booleans indicating whether each principle was applied correctly.
        """
        self.current_step += 1

        for i, data in enumerate(usage_data):
            if data is None:
                continue

            principle_id, _, content = data
            is_correct = correctness_list[i]

            if principle_id not in self.principle_stats:
                self.principle_stats[principle_id] = {
                    "score": 0.5,
                    "usage": 0,
                    "history_correct": 0,
                    "created_step": 0,
                    "is_probation": False
                }

            stats = self.principle_stats[principle_id]
            stats['usage'] += 1

            if is_correct:
                stats['history_correct'] += 1
                stats['score'] = min(1.0, stats['score'] + 0.1)
                if stats['is_probation'] and stats['score'] > 0.6:
                    stats['is_probation'] = False
            else:
                penalty = 0.2 if stats['history_correct'] <= 10 else 0.1 if stats['history_correct'] <= 50 else 0.05
                stats['score'] = max(0.0, stats['score'] - penalty)

    def prune_principles(self, min_usage=5, score_threshold=0.3):
        """
        Prune low-quality or unused principles from the database.

        Parameters:
        - min_usage (int): The minimum number of times a principle must be used to avoid pruning.
        - score_threshold (float): The score threshold below which principles are considered for pruning.

        Returns:
        - int: The number of principles that were deleted from the database.
        """
        ids_to_delete = []
        decay_rate = 0.01

        for principle_id, stats in list(self.principle_stats.items()):
            if stats['score'] < 0.95:
                stats['score'] -= decay_rate

            if stats['is_probation'] and stats['score'] < 0.4:
                ids_to_delete.append(principle_id)
                continue

            if stats['usage'] >= min_usage and stats['score'] < score_threshold:
                ids_to_delete.append(principle_id)
                continue

            age = self.current_step - stats['created_step']
            if age > 500 and stats['usage'] < 2:
                ids_to_delete.append(principle_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            for principle_id in ids_to_delete:
                del self.principle_stats[principle_id]

        return len(ids_to_delete)