import numpy as np
import re
from src.utils.embedding_loader import get_embedding_model

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    class BM25Okapi:
        """
        NumPy-optimized Pure Python fallback implementation of BM25Okapi.
        Provides robust keyword frequency statistics and Okapi term weighting.
        """
        def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
            self.k1 = k1
            self.b = b
            self.corpus_size = len(corpus)
            self.avgdl = sum(len(x) for x in corpus) / self.corpus_size if self.corpus_size > 0 else 0
            self.doc_freqs = []
            self.nd = {}
            for document in corpus:
                frequencies = {}
                for word in document:
                    frequencies[word] = frequencies.get(word, 0) + 1
                self.doc_freqs.append(frequencies)
                for word in set(document):
                    self.nd[word] = self.nd.get(word, 0) + 1
            self.idf = {}
            for word, freq in self.nd.items():
                self.idf[word] = float(np.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1))

        def get_scores(self, query: list[str]) -> np.ndarray:
            scores = np.zeros(self.corpus_size)
            for i in range(self.corpus_size):
                doc_len = sum(self.doc_freqs[i].values())
                score = 0.0
                for word in query:
                    if word in self.doc_freqs[i]:
                        freq = self.doc_freqs[i][word]
                        numerator = freq * (self.k1 + 1)
                        denom_term = self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) if self.avgdl > 0 else self.k1
                        score += self.idf.get(word, 0) * numerator / (freq + denom_term)
                scores[i] = score
            return scores

class LocalVectorDB:
    """
    A robust, extremely fast NumPy-based Vector Database.
    Avoids C-compilation and dynamic library loading crashes (like SQLite/Chroma versions on Windows)
    while providing native Cosine Similarity search over cached embeddings.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = []
        self.documents = []
        self.ids = []
        self.metadata = []

    def add_documents(self, documents: list[str], ids: list[str], metadata: list[dict]):
        model = get_embedding_model(self.model_name)
        new_embeddings = model.encode(documents, show_progress_bar=False)
        
        self.embeddings.extend(new_embeddings)
        self.documents.extend(documents)
        self.ids.extend(ids)
        self.metadata.extend(metadata)
        
    def similarity_search(self, query: str, k: int = 5) -> list[dict]:
        if not self.embeddings:
            return []
            
        model = get_embedding_model(self.model_name)
        query_emb = model.encode(query, show_progress_bar=False)
        
        # Calculate cosine similarity
        embs_arr = np.array(self.embeddings)
        norms = np.linalg.norm(embs_arr, axis=1) * np.linalg.norm(query_emb)
        dot_product = np.dot(embs_arr, query_emb)
        
        # Guard against zero-division
        scores = np.divide(dot_product, norms, out=np.zeros_like(dot_product), where=norms!=0)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            results.append({
                "id": self.ids[idx],
                "text": self.documents[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[idx])
            })
        return results

class HybridRetriever:
    """
    Implements a hybrid Dense-Sparse retrieval pipeline with Reciprocal Rank Fusion (RRF)
    and Parent Context translation.
    """
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.vector_db = LocalVectorDB(model_name=embedding_model_name)
        self.bm25 = None
        self.child_documents = []
        self.parent_map = {} # Maps parent_id -> parent text and metadata

    def index_corpus(self, parsed_corpus: dict):
        """
        Populates dense vector index and sparse BM25 index with documents.
        """
        parents = parsed_corpus.get("parents", [])
        children = parsed_corpus.get("children", [])
        
        # Save parent map for quick lookup during retrieval
        for p in parents:
            self.parent_map[p["parent_id"]] = {
                "text": p["text"],
                "metadata": p["metadata"]
            }
            
        if not children:
            print("[HybridRetriever] Warning: No children chunks found to index.")
            return

        # 1. Index dense vectors (Children)
        child_texts = [c["text"] for c in children]
        child_ids = [c["child_id"] for c in children]
        child_metadata = [c["metadata"] for c in children]
        
        print(f"[HybridRetriever] Indexing {len(child_texts)} child chunks in Dense Vector DB...")
        self.vector_db.add_documents(child_texts, child_ids, child_metadata)
        
        # 2. Index BM25 (Children)
        print(f"[HybridRetriever] Initializing BM25 index on child tokens...")
        tokenized_corpus = [self.tokenize(text) for text in child_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.child_documents = children # Hold reference to match indexes
        print("[HybridRetriever] Indexing completed successfully!")

    def tokenize(self, text: str) -> list[str]:
        """
        Simple, robust tokenization mapping words to lowercase alpha-numeric.
        """
        return re.findall(r"\w+", text.lower())

    def sparse_search(self, query: str, k: int = 5) -> list[dict]:
        """
        Performs BM25 keyword-based sparse search over child chunks.
        """
        if self.bm25 is None or not self.child_documents:
            return []
            
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_k_indices:
            score = float(scores[idx])
            if score <= 0:
                continue # Skip completely non-matching chunks
            child = self.child_documents[idx]
            results.append({
                "id": child["child_id"],
                "text": child["text"],
                "metadata": child["metadata"],
                "score": score
            })
        return results

    def reciprocal_rank_fusion(self, dense_results: list[dict], sparse_results: list[dict], 
                               k_rrf: int = 60, top_n: int = 5) -> list[dict]:
        """
        Blends Dense and Sparse ranks using Reciprocal Rank Fusion (RRF).
        RRF Score = Sum( 1 / (k_rrf + Rank_i) )
        """
        rrf_scores = {}
        doc_details = {}
        
        # Helper to process rank list
        def accumulate_rrf(results):
            for rank, doc in enumerate(results):
                doc_id = doc["id"]
                doc_details[doc_id] = doc # Store details
                
                # Accruing RRF rank metric
                score = 1.0 / (k_rrf + (rank + 1))
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + score

        accumulate_rrf(dense_results)
        accumulate_rrf(sparse_results)
        
        # Sort docs by aggregate score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_n]
        
        fused_results = []
        for doc_id in sorted_ids:
            doc = doc_details[doc_id]
            fused_results.append({
                "id": doc_id,
                "text": doc["text"],
                "metadata": doc["metadata"],
                "rrf_score": rrf_scores[doc_id]
            })
        return fused_results

    def retrieve(self, query: str, k: int = 5, top_n: int = 3) -> list[dict]:
        """
        Orchestrates full hybrid retrieval:
        1. Dense search on children
        2. Sparse search on children
        3. Reciprocal Rank Fusion blending
        4. Translating granular Child hits to rich Parent contexts
        """
        dense_hits = self.vector_db.similarity_search(query, k=k)
        sparse_hits = self.sparse_search(query, k=k)
        
        fused_hits = self.reciprocal_rank_fusion(dense_hits, sparse_hits, top_n=top_n)
        
        # Parent translation (eliminates duplication and restores holistic context)
        final_contexts = []
        seen_parents = set()
        
        for hit in fused_hits:
            parent_id = hit["metadata"].get("parent_id")
            if parent_id and parent_id in self.parent_map:
                if parent_id not in seen_parents:
                    seen_parents.add(parent_id)
                    parent_doc = self.parent_map[parent_id]
                    final_contexts.append({
                        "parent_id": parent_id,
                        "text": parent_doc["text"],
                        "metadata": parent_doc["metadata"],
                        "score": hit.get("rrf_score", 0.0)
                    })
            else:
                # Fallback to granular child text if no parent map exists
                final_contexts.append({
                    "parent_id": None,
                    "text": hit["text"],
                    "metadata": hit["metadata"],
                    "score": hit.get("rrf_score", 0.0)
                })
                
        return final_contexts
