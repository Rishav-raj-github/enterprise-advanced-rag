import json
import re
from src.config import get_llm

class QueryTransformer:
    """
    Implements advanced Query Translation techniques:
    1. Multi-Query Generation (Diverse query translations)
    2. Hypothetical Document Embeddings (HyDE)
    """
    def __init__(self):
        self.llm = get_llm()

    def generate_alternative_queries(self, original_query: str, count: int = 3) -> list[str]:
        """
        Translates a single user query into multiple diverse query perspectives
        to optimize coverage during dense/sparse lookups.
        """
        prompt = f"""You are an advanced AI retrieval orchestrator.
Your goal is to take a single user search query and output {count} distinct, highly optimized search queries that capture the user's core intent from different angles and phrasing.
Do not output any introductions or explanations. Output exactly {count} lines, each containing a single query rewrite.

Original Query: "{original_query}"

Queries:"""
        try:
            raw_output = self.llm.generate_content(prompt)
            lines = [line.strip().lstrip("0123456789.-* ") for line in raw_output.split("\n") if line.strip()]
            queries = [line for line in lines if line]
            
            # Ensure we return at least the original query if anything fails
            if not queries:
                return [original_query]
            return queries[:count]
        except Exception as e:
            print(f"[QueryTransformer Error] Multi-query generation failed: {e}")
            return [original_query]

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Implements HyDE (Hypothetical Document Embeddings).
        Generates a synthetic, high-fidelity response to serve as a symmetric
        search seed in vector retrieval.
        """
        prompt = f"""You are an expert system engineer and technical editor.
Write a concise, professional, and authoritative paragraph that directly answers the following search query.
This paragraph will be used as a search query seed to search a database of enterprise technical and policy documents.
Write ONLY the factual paragraph. Do not include introductory phrases (like "Here is the answer...") or signatures.

Search Query: "{query}"

Hypothetical Answer Document:"""
        try:
            hyde_doc = self.llm.generate_content(prompt)
            return hyde_doc.strip()
        except Exception as e:
            print(f"[QueryTransformer Error] HyDE generation failed: {e}")
            return query
