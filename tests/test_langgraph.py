import pytest
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.graph.pipeline import AdvancedRAGPipeline

def test_pipeline_execution():
    corpus_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sample_corpus")
    pipeline = AdvancedRAGPipeline(corpus_dir)
    
    # Test unstructured RAG path execution
    query = "What is the hybrid remote work stipend?"
    result = pipeline.run(query)
    
    assert "generation" in result
    assert result["generation"] is not None
    assert "node_input_guardrail" in result["execution_trace"]
    assert "node_route_query" in result["execution_trace"]
