import re
import numpy as np
from src.utils.embedding_loader import get_embedding_model

class CrossEncoderReranker:
    """
    Evaluates retrieved documents using token-to-token cross-attention.
    Provides a fallback keyword-overlap similarity model if HuggingFace/SentenceTransformers
    has downloading issues or runs out of CPU resources.
    """
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self.model = None
        self.is_active = False
        
        # Try loading the cross-encoder model
        try:
            from sentence_transformers import CrossEncoder
            print(f"[Reranker] Loading Cross-Encoder model: {model_name}...")
            self.model = CrossEncoder(model_name)
            self.is_active = True
            print("[Reranker] Cross-Encoder loaded successfully!")
        except Exception as e:
            print(f"[Reranker Warning] Could not load local cross-encoder model: {e}. Falling back to TF-IDF Overlap Similarity.")

    def compute_overlap_score(self, query: str, document: str) -> float:
        """
        Robust, zero-dependency term overlap fallback score with length normalization.
        """
        q_words = set(re.findall(r"\w+", query.lower()))
        d_words = re.findall(r"\w+", document.lower())
        d_words_set = set(d_words)
        
        if not q_words or not d_words:
            return 0.0
            
        overlap = len(q_words.intersection(d_words_set))
        # Cosine-like normalization of word bags
        score = overlap / (np.sqrt(len(q_words)) * np.sqrt(len(d_words_set)))
        return float(score)

    def rerank(self, query: str, documents: list[dict], top_n: int = 3) -> list[dict]:
        """
        Re-evaluates the relevance rank of retrieved documents relative to the query.
        """
        if not documents:
            return []
            
        reranked = []
        
        if self.is_active and self.model is not None:
            try:
                pairs = [[query, doc["text"]] for doc in documents]
                scores = self.model.predict(pairs)
                
                for idx, doc in enumerate(documents):
                    doc_copy = doc.copy()
                    doc_copy["rerank_score"] = float(scores[idx])
                    reranked.append(doc_copy)
                
                # Sort descending
                reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                print(f"[Reranker Error] Cross-Encoder inference failed: {e}. Switching to TF-IDF overlap.")
                self.is_active = False
                
        if not self.is_active:
            # Fallback
            for doc in documents:
                doc_copy = doc.copy()
                doc_copy["rerank_score"] = self.compute_overlap_score(query, doc["text"])
                reranked.append(doc_copy)
                
            reranked = sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)
            
        return reranked[:top_n]


class ContextCompressor:
    """
    Prunes retrieved documents down to only sentences containing direct answers,
    filtering noise and minimizing LLM context windows.
    """
    def __init__(self, sentences_limit: int = 4):
        self.sentences_limit = sentences_limit

    def split_into_sentences(self, text: str) -> list[str]:
        """
        Splits paragraph into logical sentences using regex.
        """
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        return [s.strip() for s in sentences if s.strip()]

    def compress(self, query: str, document_text: str) -> str:
        """
        Extracts only sentences that share terms with the search query.
        """
        sentences = self.split_into_sentences(document_text)
        q_words = set(re.findall(r"\w+", query.lower()))
        
        scored_sentences = []
        for idx, sentence in enumerate(sentences):
            s_words = set(re.findall(r"\w+", sentence.lower()))
            overlap = len(q_words.intersection(s_words))
            
            # Preserve header contexts
            if sentence.startswith("#"):
                overlap += 1
                
            scored_sentences.append((idx, sentence, overlap))
            
        # Sort sentences by overlap score descending, keeping original order for equal scores
        sorted_sentences = sorted(scored_sentences, key=lambda x: x[2], reverse=True)
        
        # Take the top N matching sentences
        top_sentences = sorted_sentences[:self.sentences_limit]
        # Sort them back in original reading order
        top_sentences = sorted(top_sentences, key=lambda x: x[0])
        
        compressed_text = " ... ".join([item[1] for item in top_sentences])
        return compressed_text
