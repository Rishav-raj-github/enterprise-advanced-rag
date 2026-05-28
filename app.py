import streamlit as st
import os
import time
import pandas as pd
import sqlite3
from src.graph.pipeline import AdvancedRAGPipeline
from src.config import SQL_DB_PATH, BASE_DIR

# Set beautiful dark/modern-themed page configurations
st.set_page_config(
    page_title="Enterprise Advanced RAG Control Center",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling rules using CSS to wow the user
st.markdown("""
<style>
    /* Custom background & typography adjustments */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    code, pre, [class*="stCode"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.9rem !important;
    }

    /* Core container cards with a premium glassmorphic aesthetic */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(8px);
        margin-bottom: 20px;
    }
    
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    
    .badge-correct { background-color: rgba(40, 167, 69, 0.15); color: #28a745; border: 1px solid #28a745; }
    .badge-ambiguous { background-color: rgba(255, 193, 7, 0.15); color: #ffc107; border: 1px solid #ffc107; }
    .badge-incorrect { background-color: rgba(220, 53, 69, 0.15); color: #dc3545; border: 1px solid #dc3545; }
    
    /* Sleek gradient main header styling */
    .gradient-text {
        background: linear-gradient(90deg, #FF4B4B, #9B51E0, #4A90E2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 10px;
    }
    
    .trace-node {
        display: inline-flex;
        align-items: center;
        background: rgba(155, 81, 224, 0.1);
        color: #b182ff;
        border: 1px solid rgba(155, 81, 224, 0.3);
        padding: 6px 14px;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 4px;
    }
</style>
""", unsafe_allow_allowed_html=True, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Sidebar and App Configurations
# ------------------------------------------------------------------------------
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ Operations Console</h2>", unsafe_allow_html=True)

# Ingest directories
corpus_dir = os.path.join(BASE_DIR, "data", "sample_corpus")

# Initialize and Cache the Pipeline engine globally
@st.cache_resource(show_spinner=True)
def get_pipeline():
    return AdvancedRAGPipeline(corpus_dir)

with st.spinner("Initializing Enterprise Advanced RAG Cognitive State Graph Engine..."):
    pipeline = get_pipeline()

st.sidebar.success("LangGraph state graph engine online!")

# Quick stats in the sidebar
st.sidebar.markdown("### 📊 Dataset Inventory")
num_docs = len(os.listdir(corpus_dir)) if os.path.exists(corpus_dir) else 0
st.sidebar.metric(label="Ingested Policy & Tech Documents", value=f"{num_docs} Files")

# Database inspect tools in sidebar
st.sidebar.markdown("### 🗃️ SQLite Seeds Inspector")
if st.sidebar.button("Fetch Relational SQLite Tables"):
    try:
        conn = sqlite3.connect(SQL_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [r[0] for r in cursor.fetchall() if r[0] != "sqlite_sequence"]
        conn.close()
        st.sidebar.write("Available Database Tables:")
        st.sidebar.json(tables)
    except Exception as e:
        st.sidebar.error(f"Error accessing DB: {e}")

# ------------------------------------------------------------------------------
# Main Dashboard Layout
# ------------------------------------------------------------------------------
st.markdown("<div class='gradient-text'>🧬 Enterprise Advanced RAG Core</div>", unsafe_allow_html=True)
st.markdown("<p style='font-size:1.15rem; color:#aaa; margin-top:-10px; margin-bottom: 25px;'>Stateful Cognitive Architecture featuring Hybrid Search, RRF, ReRanking, HyDE, CRAG, Self-RAG, Text2SQL and Semantic Caching.</p>", unsafe_allow_html=True)

query = st.text_input("💬 Ask the Platform (e.g. stipend details, cloud encryption, database sales, or write threat strings):", 
                      value="What are the remote work equipment stipend rules for Acme employees?")

if st.button("🚀 Process Query through LangGraph State Machine", type="primary"):
    start_time = time.time()
    
    with st.spinner("Invoking LangGraph pipeline nodes..."):
        # Run through state graph
        result = pipeline.run(query)
        
    execution_time = time.time() - start_time
    
    # 1. Metric Indicators Panel
    st.markdown("### ⏱️ Latency & Pipeline Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Total Latency", value=f"{execution_time:.3f} s")
    with col2:
        cache_text = "HIT (Sub-ms)" if result.get("cache_hit") else "MISS"
        st.metric(label="Semantic Cache Result", value=cache_text)
    with col3:
        routing = result.get("routing_decision", "unstructured").upper()
        st.metric(label="Active Pipeline Path", value=routing)
    with col4:
        loops = result.get("loop_count", 0)
        st.metric(label="Self-RAG Loops executed", value=f"{loops} Cycles")

    st.markdown("---")

    # 2. Execution Tracing Flow Panel
    st.markdown("### ⛓️ Real-Time LangGraph Node-by-Node Execution Trace")
    trace = result.get("execution_trace", [])
    if trace:
        html_trace = []
        for i, node in enumerate(trace):
            html_trace.append(f"<span class='trace-node'>{node}</span>")
            if i < len(trace) - 1:
                html_trace.append("<span style='font-size:1.2rem; font-weight:bold; color:#FF4B4B;'>➔</span>")
        st.markdown(f"<div style='margin-bottom:25px;'>{' '.join(html_trace)}</div>", unsafe_allow_html=True)
    else:
        st.write("No execution trace logs generated.")

    # 3. Main Result Output Panel
    st.markdown("### 📖 Corporate Synthesized Response")
    st.markdown(f"<div class='glass-card' style='border-left: 5px solid #9B51E0;'>{result['generation']}</div>", unsafe_allow_html=True)

    # 4. Context Inspection Panel
    if result.get("routing_decision") == "structured" and result.get("sql_query"):
        st.markdown("### 💾 Structured Query Execution Details")
        st.code(result.get("sql_query"), language="sql")
        
        sql_data = result.get("sql_data", {})
        if sql_data.get("success"):
            st.markdown("#### Database Output:")
            st.dataframe(pd.DataFrame(sql_data["records"]), use_container_width=True)
        else:
            st.error(f"SQL Execution Failed: {sql_data.get('error')}")
            
    elif result.get("retrieved_docs"):
        st.markdown("### 🔍 Retrieved Document context & Corrective RAG (CRAG) Grades")
        
        for idx, doc in enumerate(result["retrieved_docs"]):
            grade = doc.get("grade", "CORRECT").upper()
            badge_class = "badge-correct"
            if grade == "AMBIGUOUS":
                badge_class = "badge-ambiguous"
            elif grade == "INCORRECT":
                badge_class = "badge-incorrect"
                
            with st.expander(f"Document {idx+1}: {doc['metadata'].get('source', 'Unknown')} (RRF Rank Score: {doc.get('score', 0.0):.5f})"):
                st.markdown(f"**CRAG Evaluator Relevance:** <span class='metric-badge {badge_class}'>{grade}</span>", unsafe_allow_html=True)
                st.markdown(f"**Chunk Payloads:**\n\n{doc['text']}")
                st.json(doc["metadata"])

# ------------------------------------------------------------------------------
# Bottom Educational Documentation Panels
# ------------------------------------------------------------------------------
st.markdown("<br><br>", unsafe_allow_html=True)
with st.expander("📚 Explore Academic Foundations of the Pipeline"):
    st.markdown("""
    This control dashboard maps directly to the educational notebooks generated inside the `/notebooks` folder:
    
    *   **01_data_ingestion_hybrid_search**: Sets up Parent-Child splitting, maps Child vectors to Parent context blocks, and implements RRF Rank Blending.
    *   **02_query_transformation_hyde**: Generates multi-query synonyms and hypothetical paragraphs to bridge query-document asymmetry.
    *   **03_reranking_context_compression**: Cross-Encoder attention re-ranks and filters irrelevant sentences.
    *   **04_corrective_rag_crag**: Grading docs as Correct/Incorrect/Ambiguous, triggering fallback web-searches where relevance ratio drops.
    *   **05_self_rag**: Implements Self-RAG loop verifying Faithfulness and Utility before output delivery.
    *   **06_structured_text2sql**: Reflects sqlite schemas, translates questions to SELECT strings and executes table transactions.
    *   **07_caching_and_guardrails**: Blocks injection threats and speeds up duplicate prompts with a sub-millisecond SQLite Cosine similarity Cache.
    *   **08_langgraph_orchestrator**: Compiles Graph nodes and orchestrates all layers in a stateful cyclic system.
    """)
