try:
    from langgraph.graph import StateGraph, END
except ImportError:
    # A beautiful custom state machine engine that runs identical cyclic state transitions
    END = "__end__"
    
    class CompiledGraph:
        def __init__(self, nodes, edges, conditional_edges, entry_point):
            self.nodes = nodes
            self.edges = edges
            self.conditional_edges = conditional_edges
            self.entry_point = entry_point
            
        def invoke(self, state):
            current_node = self.entry_point
            print(f"[State Machine Fallback] Kicking off compiled state transitions at entry point: '{current_node}'")
            
            # Cyclic execution loop
            visited = set()
            while current_node != END:
                # 1. Execute Node function
                node_func = self.nodes[current_node]
                state = node_func(state)
                
                # Prevent infinite cycles in faulty conditions
                visited_key = (current_node, state.get("loop_count", 0))
                if visited_key in visited:
                    print(f"[State Machine Fallback] Detected redundant cycle loop at node '{current_node}'. Halting execution.")
                    break
                visited.add(visited_key)
                
                # 2. Find next transition
                next_node = None
                
                # Check conditional edges first
                if current_node in self.conditional_edges:
                    routing_func, path_map = self.conditional_edges[current_node]
                    decision = routing_func(state)
                    next_node = path_map.get(decision)
                    print(f"[State Machine Fallback] Conditional Edge routing at '{current_node}' via decision '{decision}' -> '{next_node}'")
                
                # Check static edges
                elif current_node in self.edges:
                    next_node = self.edges[current_node]
                    print(f"[State Machine Fallback] Static Edge transition from '{current_node}' -> '{next_node}'")
                    
                if not next_node or (next_node not in self.nodes and next_node != END):
                    print(f"[State Machine Fallback] Error: Next node '{next_node}' not found. Terminating.")
                    break
                    
                current_node = next_node
                
            return state

    class StateGraph:
        def __init__(self, state_schema):
            self.state_schema = state_schema
            self.nodes = {}
            self.edges = {}
            self.conditional_edges = {}
            self.entry_point = None
            print("[State Machine Fallback] Initializing zero-dependency StateGraph compiler fallback.")

        def add_node(self, name, func):
            self.nodes[name] = func

        def add_edge(self, source, target):
            self.edges[source] = target

        def add_conditional_edges(self, source, routing_func, path_map):
            self.conditional_edges[source] = (routing_func, path_map)

        def set_entry_point(self, name):
            self.entry_point = name

        def compile(self):
            return CompiledGraph(self.nodes, self.edges, self.conditional_edges, self.entry_point)
from src.graph.state import GraphState
from src.config import get_llm, SQL_DB_PATH
from src.ingestion.chunking import load_and_chunk_corpus
from src.retrieval.hybrid import HybridRetriever
from src.transformation.hyde import QueryTransformer
from src.ranking.reranker import CrossEncoderReranker, ContextCompressor
from src.agents.crag_evaluator import DocumentGrader
from src.agents.self_rag_evaluator import SelfRAGGrader
from src.agents.text2sql import Text2SQLAgent
from src.guardrails.validator import EnterpriseGuardrails
from src.cache.semantic_cache import SemanticCache

import os

class AdvancedRAGPipeline:
    """
    Compiles and manages the cyclic, stateful LangGraph Enterprise Advanced RAG pipeline.
    Instantiates all underlying agent evaluators, retrievers, caches, and guardrails.
    """
    def __init__(self, corpus_dir: str):
        self.corpus_dir = corpus_dir
        self.llm = get_llm()
        
        # Initialize sub-systems
        self.retriever = HybridRetriever()
        self.transformer = QueryTransformer()
        self.reranker = CrossEncoderReranker()
        self.compressor = ContextCompressor()
        self.doc_grader = DocumentGrader()
        self.self_rag_grader = SelfRAGGrader()
        self.sql_agent = Text2SQLAgent(SQL_DB_PATH)
        self.guardrails = EnterpriseGuardrails()
        self.cache = SemanticCache()
        
        # Automatically ingest and index the unstructured corpus
        self._index_corpus()
        
        # Assemble LangGraph Workflow
        self.workflow = self._compile_graph()

    def _index_corpus(self):
        print(f"[Pipeline Ingestion] Loading corpus from {self.corpus_dir}...")
        parsed_corpus = load_and_chunk_corpus(self.corpus_dir)
        self.retriever.index_corpus(parsed_corpus)

    # ==========================================================================
    # Graph Nodes Implementation
    # ==========================================================================

    def node_input_guardrail(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_input_guardrail"]
        query = state["query"]
        
        result = self.guardrails.validate_input(query)
        if not result["is_safe"]:
            return {
                **state,
                "safety_triggered": True,
                "safety_details": result["reason"],
                "generation": "SYSTEM ERROR: The query has been flagged by enterprise safety guardrails.",
                "execution_trace": trace
            }
            
        return {
            **state,
            "safety_triggered": False,
            "execution_trace": trace
        }

    def node_route_query(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_route_query"]
        query = state["query"].lower()
        
        # Direct semantic heuristic routing
        structured_terms = [
            "sales", "revenue", "price", "highest price", "customer", 
            "product", "quantities", "transaction", "purchased", "table"
        ]
        
        decision = "unstructured"
        for term in structured_terms:
            if term in query:
                decision = "structured"
                break
                
        print(f"[Router Node] Routing query intent -> {decision.upper()}")
        return {
            **state,
            "routing_decision": decision,
            "execution_trace": trace
        }

    def node_query_translate(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_query_translate"]
        query = state["query"]
        
        # Execute Multi-Query expansion
        alternative_queries = self.transformer.generate_alternative_queries(query, count=2)
        # Execute HyDE generation
        hyde_doc = self.transformer.generate_hypothetical_document(query)
        
        # Combine expanded terms into state
        trace.append(f"expanded_queries: {len(alternative_queries)}")
        return {
            **state,
            "query": f"{query} | Alternative: {' '.join(alternative_queries)} | Hypothetical: {hyde_doc}",
            "execution_trace": trace
        }

    def node_retrieve(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_retrieve"]
        expanded_query = state["query"]
        original_query = state["original_query"]
        
        # Perform retrieval using hybrid RRF
        docs = self.retriever.retrieve(original_query, k=5, top_n=4)
        
        return {
            **state,
            "retrieved_docs": docs,
            "execution_trace": trace
        }

    def node_rerank_compress(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_rerank_compress"]
        original_query = state["original_query"]
        docs = state["retrieved_docs"]
        
        if not docs:
            return {**state, "execution_trace": trace}
            
        # 1. Re-evaluate rank via Cross-Encoder
        reranked = self.reranker.rerank(original_query, docs, top_n=3)
        
        # 2. Compress sentence payloads
        compressed_docs = []
        for doc in reranked:
            compressed_text = self.compressor.compress(original_query, doc["text"])
            doc_copy = doc.copy()
            doc_copy["text"] = compressed_text
            compressed_docs.append(doc_copy)
            
        return {
            **state,
            "retrieved_docs": compressed_docs,
            "execution_trace": trace
        }

    def node_grade_docs(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_grade_docs"]
        original_query = state["original_query"]
        docs = state["retrieved_docs"]
        
        evaluation = self.doc_grader.evaluate_retrievals(original_query, docs)
        
        print(f"[CRAG Node] Search Fallback Flagged -> {evaluation['search_fallback']}")
        return {
            **state,
            "retrieved_docs": evaluation["graded_documents"],
            "search_fallback": evaluation["search_fallback"],
            "execution_trace": trace
        }

    def node_web_search(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_web_search"]
        original_query = state["original_query"]
        docs = state["retrieved_docs"]
        
        # Simulate Tavily/Web Search
        web_results = []
        print(f"[Web Search Fallback] Triggered external lookup for: '{original_query}'")
        
        # Mock structured response from web search API (highly relevant details)
        if "leave" in original_query or "vacation" in original_query:
            web_results.append({
                "parent_id": "web_search_1",
                "text": "Acme Corp provides PTO accrued monthly. Unused PTO up to 5 days rolls over to next year, standard HR manual section 2.1.",
                "metadata": {"source": "web_search_api", "type": "web_fallback"},
                "score": 0.95
            })
        elif "revenue" in original_query or "sales" in original_query:
            web_results.append({
                "parent_id": "web_search_2",
                "text": "Seeded SQLite logs show sales transactions totaling over 200 records in products, hardware, software, and cloud categories.",
                "metadata": {"source": "web_search_api", "type": "web_fallback"},
                "score": 0.95
            })
        else:
            web_results.append({
                "parent_id": "web_search_general",
                "text": f"Simulated Web Search results matching keyword analysis for: {original_query}. Detailed logs indicate consistent server mesh uptime.",
                "metadata": {"source": "web_search_api", "type": "web_fallback"},
                "score": 0.85
            })
            
        # Merge web results with current contexts
        merged_docs = docs + web_results
        
        return {
            **state,
            "retrieved_docs": merged_docs,
            "search_fallback": False, # Fallback satisfied
            "execution_trace": trace
        }

    def node_text_to_sql(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_text_to_sql"]
        original_query = state["original_query"]
        
        sql_result = self.sql_agent.process(original_query)
        
        return {
            **state,
            "sql_query": sql_result.get("sql"),
            "sql_data": sql_result,
            "execution_trace": trace
        }

    def node_generate(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_generate"]
        original_query = state["original_query"]
        routing = state["routing_decision"]
        loop_count = state.get("loop_count", 0) + 1
        
        # Assemble context blocks depending on routing path
        if routing == "structured":
            sql_data = state.get("sql_data", {})
            if sql_data.get("success"):
                context = f"SQL Query Run: {sql_data['sql']}\n\nData Result:\n{sql_data['markdown']}"
            else:
                context = f"SQL Error Encountered: {sql_data.get('error')}"
        else:
            docs = state.get("retrieved_docs", [])
            context = "\n\n".join([f"Source: {d['metadata']['source']}\nContent: {d['text']}" for d in docs])
            
        prompt = f"""You are an elite corporate assistant at Acme Corp.
Answer the user's question clearly, professionally, and in complete details based ONLY on the provided system context.
If the context doesn't provide enough details, state that clearly rather than fabricating facts.
Always synthesize structured summaries and highlight critical items in tables or bullets where appropriate.

System Context:
{context}

User Question: "{original_query}"

Corporate Answer:"""
        try:
            generation = self.llm.generate_content(prompt).strip()
        except Exception as e:
            generation = f"Generation failed due to: {e}"
            
        return {
            **state,
            "generation": generation,
            "loop_count": loop_count,
            "execution_trace": trace
        }

    def node_self_rag_eval(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_self_rag_eval"]
        original_query = state["original_query"]
        generation = state["generation"]
        docs = state["retrieved_docs"]
        
        context = "\n".join([d["text"] for d in docs])
        
        evaluation = self.self_rag_grader.evaluate_response(generation, context, original_query)
        
        print(f"[Self-RAG Node] Answer Accepted? -> {evaluation['is_accepted']}")
        return {
            **state,
            "faithfulness_grade": evaluation["faithfulness"],
            "utility_grade": evaluation["utility"],
            "execution_trace": trace
        }

    def node_block_safety(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_block_safety"]
        return {
            **state,
            "generation": f"SECURITY BLOCK: The query '{state['original_query']}' was flag-blocked by system threat-detection. Safety details: {state.get('safety_details')}",
            "execution_trace": trace
        }

    def node_set_cache(self, state: GraphState) -> GraphState:
        trace = state.get("execution_trace", []) + ["node_set_cache"]
        original_query = state["original_query"]
        generation = state["generation"]
        
        # Update semantic cache for fast future serving
        if not state.get("safety_triggered") and not state.get("cache_hit"):
            self.cache.set_cache(original_query, generation)
            
        return {
            **state,
            "execution_trace": trace
        }

    # ==========================================================================
    # Graph Routing Logic (Conditional Edges)
    # ==========================================================================

    def edge_check_guardrail(self, state: GraphState) -> str:
        if state.get("safety_triggered"):
            return "blocked"
        return "passed"

    def edge_router(self, state: GraphState) -> str:
        return state["routing_decision"]

    def edge_crag(self, state: GraphState) -> str:
        if state.get("search_fallback"):
            return "fallback"
        return "generate"

    def edge_self_rag(self, state: GraphState) -> str:
        # Check if faithfulness or utility failed, and we have loop headroom
        if state.get("faithfulness_grade") == "HALLUCINATED" or state.get("utility_grade") == "USELESS":
            if state.get("loop_count", 0) < 3:
                print(f"[Self-RAG Loop] Quality check failed (Loop {state['loop_count']}/3). Retrying search fallback.")
                return "retry"
        return "cache"

    # ==========================================================================
    # Workflow Compilation
    # ==========================================================================

    def _compile_graph(self):
        graph = StateGraph(GraphState)
        
        # 1. Add Nodes
        graph.add_node("guardrail", self.node_input_guardrail)
        graph.add_node("route", self.node_route_query)
        graph.add_node("query_translate", self.node_query_translate)
        graph.add_node("retrieve", self.node_retrieve)
        graph.add_node("rerank_compress", self.node_rerank_compress)
        graph.add_node("grade_docs", self.node_grade_docs)
        graph.add_node("web_search", self.node_web_search)
        graph.add_node("text_to_sql", self.node_text_to_sql)
        graph.add_node("generate", self.node_generate)
        graph.add_node("self_rag_eval", self.node_self_rag_eval)
        graph.add_node("block_safety", self.node_block_safety)
        graph.add_node("set_cache", self.node_set_cache)
        
        # 2. Define Flow Transitions (Edges)
        graph.set_entry_point("guardrail")
        
        graph.add_conditional_edges(
            "guardrail",
            self.edge_check_guardrail,
            {
                "blocked": "block_safety",
                "passed": "route"
            }
        )
        
        graph.add_conditional_edges(
            "route",
            self.edge_router,
            {
                "structured": "text_to_sql",
                "unstructured": "query_translate"
            }
        )
        
        # Unstructured Path Flow
        graph.add_edge("query_translate", "retrieve")
        graph.add_edge("retrieve", "rerank_compress")
        graph.add_edge("rerank_compress", "grade_docs")
        
        graph.add_conditional_edges(
            "grade_docs",
            self.edge_crag,
            {
                "fallback": "web_search",
                "generate": "generate"
            }
        )
        
        graph.add_edge("web_search", "generate")
        
        # Structured Path Flow
        graph.add_edge("text_to_sql", "generate")
        
        # Post Generation Evaluation & Loops
        graph.add_edge("generate", "self_rag_eval")
        
        graph.add_conditional_edges(
            "self_rag_eval",
            self.edge_self_rag,
            {
                "retry": "web_search",
                "cache": "set_cache"
            }
        )
        
        graph.add_edge("set_cache", END)
        graph.add_edge("block_safety", END)
        
        return graph.compile()

    def run(self, query: str) -> dict:
        """
        Executes the entire RAG pipeline from a single entrypoint query.
        Includes Semantic Cache check before kicking off LangGraph thread execution.
        """
        print(f"\n[Incoming Query] '{query}'")
        
        # 1. Semantic Cache check
        cache_hit, response, score = self.cache.check_cache(query)
        if cache_hit:
            return {
                "query": query,
                "generation": response,
                "cache_hit": True,
                "execution_trace": ["semantic_cache_hit"],
                "routing_decision": "cache",
                "sql_query": None,
                "retrieved_docs": []
            }
            
        # 2. LangGraph Execution
        initial_state: GraphState = {
            "query": query,
            "original_query": query,
            "routing_decision": None,
            "retrieved_docs": [],
            "generation": None,
            "sql_query": None,
            "sql_data": None,
            "search_fallback": False,
            "loop_count": 0,
            "faithfulness_grade": None,
            "utility_grade": None,
            "safety_triggered": False,
            "safety_details": None,
            "cache_hit": False,
            "execution_trace": []
        }
        
        final_state = self.workflow.invoke(initial_state)
        return final_state
