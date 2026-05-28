from typing import TypedDict, List, Dict, Any, Optional

class GraphState(TypedDict):
    """
    State definition for the cyclic LangGraph workflow.
    Ensures structured type enforcement across all pipeline nodes.
    """
    query: str
    original_query: str
    routing_decision: Optional[str]            # "structured" or "unstructured"
    retrieved_docs: List[Dict[str, Any]]
    generation: Optional[str]
    sql_query: Optional[str]
    sql_data: Optional[Dict[str, Any]]
    search_fallback: bool
    loop_count: int
    faithfulness_grade: Optional[str]         # "FAITHFUL" or "HALLUCINATED"
    utility_grade: Optional[str]              # "HIGHLY_USEFUL", "MARGINAL", "USELESS"
    safety_triggered: bool
    safety_details: Optional[str]
    cache_hit: bool
    execution_trace: List[str]                 # Node-by-node trace logger
