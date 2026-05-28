import sqlite3
import json
import numpy as np
import datetime
from src.config import BASE_DIR, SEMANTIC_CACHE_THRESHOLD
from src.utils.embedding_loader import get_embedding_model

class SemanticCache:
    """
    State-of-the-art SQLite & NumPy semantic cache.
    Stores query-response records, embedding vector lists, and serves instant
    responses for queries sharing >= threshold similarity, slashing LLM latencies.
    """
    def __init__(self, db_name: str = "semantic_cache.db", threshold: float = SEMANTIC_CACHE_THRESHOLD):
        self.db_path = os.path.join(BASE_DIR, "data", db_name)
        self.threshold = threshold
        self.embedding_model_name = "all-MiniLM-L6-v2"
        self._init_db()

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cache_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            embedding TEXT NOT NULL, -- JSON serialized float list
            response TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
        """)
        conn.commit()
        conn.close()

    def check_cache(self, query: str) -> tuple[bool, str | None, float]:
        """
        Computes embedding for incoming query, evaluates cosine similarity
        against cache history, and returns the response if similarity exceeds threshold.
        
        Returns:
            (cache_hit_bool, response_text, similarity_score)
        """
        # Load embedding model
        model = get_embedding_model(self.embedding_model_name)
        query_emb = np.array(model.encode(query, show_progress_bar=False))
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT query, embedding, response FROM cache_store")
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return False, None, 0.0
            
        best_score = 0.0
        best_response = None
        
        # Vectorized cosine similarity computation across cached entries
        for cached_query, emb_str, response in rows:
            try:
                cached_emb = np.array(json.loads(emb_str))
                
                # Math check: Cosine Similarity
                norm_prod = np.linalg.norm(query_emb) * np.linalg.norm(cached_emb)
                if norm_prod == 0:
                    continue
                score = np.dot(query_emb, cached_emb) / norm_prod
                
                if score > best_score:
                    best_score = float(score)
                    best_response = response
            except Exception as e:
                print(f"[SemanticCache Error] Failed parsing entry: {e}")
                continue
                
        if best_score >= self.threshold:
            print(f"[SemanticCache Hit] Cosine Similarity: {best_score:.4f} >= Threshold: {self.threshold}")
            return True, best_response, best_score
            
        return False, None, best_score

    def set_cache(self, query: str, response: str):
        """
        Persists a query-response translation and its embedding in SQLite.
        """
        try:
            model = get_embedding_model(self.embedding_model_name)
            query_emb = model.encode(query, show_progress_bar=False).tolist()
            emb_str = json.dumps(query_emb)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO cache_store (query, embedding, response, timestamp) VALUES (?, ?, ?, ?)",
                (query, emb_str, response, datetime.datetime.utcnow().isoformat())
            )
            conn.commit()
            conn.close()
            print("[SemanticCache Write] Cache updated successfully!")
        except Exception as e:
            print(f"[SemanticCache Error] Failed writing cache: {e}")
import os # Explicit import for internal os usage inside class
