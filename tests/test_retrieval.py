import pytest
import os
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retrieval.hybrid import LocalVectorDB, HybridRetriever
from src.ingestion.chunking import ParentChildChunker

def test_local_vector_db():
    db = LocalVectorDB()
    
    docs = ["This is a test document about artificial intelligence.", "This is a document about soccer."]
    ids = ["doc1", "doc2"]
    metas = [{"source": "test1"}, {"source": "test2"}]
    
    db.add_documents(docs, ids, metas)
    results = db.similarity_search("artificial intelligence", k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
    assert "artificial" in results[0]["text"]

def test_hybrid_rrf_blending():
    retriever = HybridRetriever()
    
    # Mock documents for RRF ranking
    dense_results = [
        {"id": "doc_a", "text": "Stipends are 1500.", "metadata": {"parent_id": "parent_x"}},
        {"id": "doc_b", "text": "Leave carries over.", "metadata": {"parent_id": "parent_y"}}
    ]
    sparse_results = [
        {"id": "doc_b", "text": "Leave carries over.", "metadata": {"parent_id": "parent_y"}},
        {"id": "doc_a", "text": "Stipends are 1500.", "metadata": {"parent_id": "parent_x"}}
    ]
    
    # Blending RRF
    blended = retriever.reciprocal_rank_fusion(dense_results, sparse_results, top_n=2)
    assert len(blended) == 2
    assert blended[0]["id"] in ["doc_a", "doc_b"]
