from src.config import get_llm

class DocumentGrader:
    """
    Implements the Corrective RAG (CRAG) document evaluator.
    Grades retrieved chunks for relevance to determine if web search fallback is needed.
    """
    def __init__(self):
        self.llm = get_llm()

    def grade_document(self, query: str, document_text: str) -> str:
        """
        Grades a single document's relevance against the query.
        Returns one of: 'CORRECT', 'INCORRECT', 'AMBIGUOUS'
        """
        prompt = f"""You are an expert quality assurance systems agent.
Analyze the provided document context and evaluate whether it contains direct, relevant information to answer the search query.

Grade criteria:
- Output "CORRECT" if the document has highly relevant details that address the query.
- Output "AMBIGUOUS" if the document contains related topics but lacks precise answers.
- Output "INCORRECT" if the document is completely irrelevant to the query.

Strict formatting rule: You must output ONLY one word representing the grade, with no other text, details, or reasoning.

Search Query: "{query}"

Document Context:
"{document_text}"

Grade:"""
        try:
            grade = self.llm.generate_content(prompt).strip().upper()
            
            # Clean response (sometimes models add quotes or punctuation)
            for clean_word in ["CORRECT", "INCORRECT", "AMBIGUOUS"]:
                if clean_word in grade:
                    return clean_word
            
            return "AMBIGUOUS" # Fallback if model output is unstructured
        except Exception as e:
            print(f"[DocumentGrader Error] Grading failed: {e}")
            return "AMBIGUOUS"

    def evaluate_retrievals(self, query: str, documents: list[dict], relevance_threshold: float = 0.5) -> dict:
        """
        Grades all documents in list. Returns detailed metrics and 
        sets the search_fallback boolean flag if overall relevance is insufficient.
        """
        graded_docs = []
        correct_count = 0
        ambiguous_count = 0
        incorrect_count = 0
        
        for doc in documents:
            grade = self.grade_document(query, doc["text"])
            doc_copy = doc.copy()
            doc_copy["grade"] = grade
            graded_docs.append(doc_copy)
            
            if grade == "CORRECT":
                correct_count += 1
            elif grade == "AMBIGUOUS":
                ambiguous_count += 1
            else:
                incorrect_count += 1
                
        total = len(documents)
        if total == 0:
            return {
                "graded_documents": [],
                "search_fallback": True,
                "relevance_ratio": 0.0
            }
            
        # Calculate ratio of helpful documents
        # CORRECT = 1.0 weight, AMBIGUOUS = 0.5 weight, INCORRECT = 0.0 weight
        relevance_score = (correct_count * 1.0 + ambiguous_count * 0.5) / total
        search_fallback = relevance_score < relevance_threshold
        
        return {
            "graded_documents": graded_docs,
            "search_fallback": search_fallback,
            "relevance_ratio": relevance_score,
            "metrics": {
                "correct": correct_count,
                "ambiguous": ambiguous_count,
                "incorrect": incorrect_count
            }
        }
