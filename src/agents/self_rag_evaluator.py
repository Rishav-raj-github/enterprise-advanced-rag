from src.config import get_llm

class SelfRAGGrader:
    """
    Implements the Self-RAG reflection layer.
    Performs dual grading:
    1. Faithfulness (checks for model hallucination against retrieval context)
    2. Utility (checks for answer relevance against the original user query)
    """
    def __init__(self):
        self.llm = get_llm()

    def grade_faithfulness(self, response: str, context: str) -> str:
        """
        Grades whether the response is fully grounded in the provided context
        without introducing fabricated facts (hallucinations).
        Returns: 'FAITHFUL' or 'HALLUCINATED'
        """
        prompt = f"""You are an audit agent verifying factual truth.
Verify whether the generated response is fully and completely supported by the provided database context.
If the response mentions facts, numbers, or rules NOT explicitly documented in the context, you must mark it as hallucinated.

Grade criteria:
- Output "FAITHFUL" if every single claim in the response is explicitly grounded in the database context.
- Output "HALLUCINATED" if the response introduces facts, numbers, configurations, or claims that are missing from the context.

Strict formatting rule: You must output ONLY one word representing the grade, with no other text, details, or reasoning.

Response to Audit:
"{response}"

Ground Truth Context:
"{context}"

Grade:"""
        try:
            grade = self.llm.generate_content(prompt).strip().upper()
            if "FAITHFUL" in grade:
                return "FAITHFUL"
            if "HALLUCINATED" in grade:
                return "HALLUCINATED"
            return "FAITHFUL" # Resilient fallback
        except Exception as e:
            print(f"[SelfRAGGrader Error] Faithfulness grading failed: {e}")
            return "FAITHFUL"

    def grade_utility(self, response: str, query: str) -> str:
        """
        Grades whether the response is highly useful and fully answers
        the user's initial query.
        Returns: 'HIGHLY_USEFUL', 'MARGINAL', or 'USELESS'
        """
        prompt = f"""You are a customer satisfaction auditor.
Verify whether the generated response directly, completely, and effectively answers the search query.

Grade criteria:
- Output "HIGHLY_USEFUL" if the response directly addresses the query and provides a clear, actionable answer.
- Output "MARGINAL" if the response is partially helpful but misses key aspects of the query or is too generic.
- Output "USELESS" if the response fails to answer the query or says it cannot help.

Strict formatting rule: You must output ONLY one word representing the grade, with no other text, details, or reasoning.

User Query: "{query}"

Generated Response:
"{response}"

Grade:"""
        try:
            grade = self.llm.generate_content(prompt).strip().upper()
            for clean_word in ["HIGHLY_USEFUL", "MARGINAL", "USELESS"]:
                if clean_word in grade:
                    return clean_word
            return "HIGHLY_USEFUL"
        except Exception as e:
            print(f"[SelfRAGGrader Error] Utility grading failed: {e}")
            return "HIGHLY_USEFUL"

    def evaluate_response(self, response: str, context: str, query: str) -> dict:
        """
        Runs both checks and determines if the generation is accepted
        or requires looping back for regeneration.
        """
        faithfulness = self.grade_faithfulness(response, context)
        utility = self.grade_utility(response, query)
        
        # Determine overall quality state
        is_safe = (faithfulness == "FAITHFUL") and (utility in ["HIGHLY_USEFUL", "MARGINAL"])
        
        return {
            "faithfulness": faithfulness,
            "utility": utility,
            "is_accepted": is_safe,
            "action": "PASS" if is_safe else "REGENERATE"
        }
