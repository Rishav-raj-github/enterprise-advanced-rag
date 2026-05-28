# 🧬 Enterprise Advanced RAG in LangGraph

An enterprise-grade, state-of-the-art **Retrieval-Augmented Generation (RAG)** platform designed for production-scale, highly-secure cognitive search and analytics. Choreographed by a cyclic state machine built on **LangGraph**, this pipeline seamlessly blends structured **Text-to-SQL database execution** with unstructured **parent-child vector retrieval**, reinforced by re-ranking, Hypothetical Document Embeddings (HyDE), Corrective RAG (CRAG) grader nodes, Self-RAG reflection loops, semantic caching, and strict security guardrails.

---

## 🗺️ System Architecture

Below is the state-driven cognitive architecture compiled and orchestrated inside LangGraph:

```mermaid
graph TD
    UserQuery([User Input]) --> CacheCheck{Semantic Cache?}
    
    CacheCheck -- Hit (Sub-ms) --> GuardrailsOut[Output Guardrails]
    CacheCheck -- Miss --> GuardrailsIn[Input Guardrails]
    
    GuardrailsIn -- Block --> SafetyBlock[Safety Policy Response]
    GuardrailsIn -- Pass --> RouteQuery{Query Router}
    
    RouteQuery -- Unstructured Query --> QueryTranslate[Query Translation & HyDE]
    RouteQuery -- Structured Query --> Text2SQL[SQLite Text2SQL Agent]
    
    QueryTranslate --> HybridSearch[Hybrid Search: Dense Vector + BM25 Sparse]
    HybridSearch --> RRF[Reciprocal Rank Fusion]
    RRF --> ReRanking[Cross-Encoder ReRanking]
    ReRanking --> ContextCompress[Context Compression & Summarization]
    ContextCompress --> CRAG{Corrective RAG Evaluator}
    
    CRAG -- Fully Relevant --> SelfRAG[Self-RAG Generator]
    CRAG -- Marginally Relevant / Gaps --> WebSearch[Fallback Search: Tavily API]
    CRAG -- Irrelevant --> WebSearch
    
    WebSearch --> SelfRAG
    
    Text2SQL --> SQLiteQuery[Execute SQLite DB Query]
    SQLiteQuery --> SQLFormat[Synthesize Tabular Result]
    SQLFormat --> SelfRAG
    
    SelfRAG --> HallucinationGrade{Hallucination Checker}
    
    HallucinationGrade -- Hallucinated --> RegenerateSelfRAG[Re-Query / Re-Generate]
    HallucinationGrade -- Faithful --> UtilityGrade{Utility / Answer Relevance Grade}
    
    UtilityGrade -- Irrelevant/Poor --> WebSearch
    UtilityGrade -- Highly Useful --> SetCache[Write to Semantic Cache]
    
    SetCache --> GuardrailsOut
    SafetyBlock --> UserOutput([System Output])
    GuardrailsOut -- Safe --> UserOutput
    GuardrailsOut -- Flagged --> SafetyBlock
```

---

## ✨ Features & Component Design

1. **Stateful Graph Orchestration (`src/graph/`)**: A cyclical state machine defining clear nodes (`retrieve`, `grade_docs`, `text_to_sql`, `generate`) and conditional routing edges compiled with error-resilience.
2. **Hybrid Retrieval with Reciprocal Rank Fusion (`src/retrieval/`)**: Melds semantic Dense Vector similarity and lexical BM25 token frequencies using **RRF** scores to capture both semantic context and exact word matches:
   $$\text{RRF Score}(d) = \sum_{m \in \text{Retrievers}} \frac{1}{k_{rrf} + r_m(d)}$$
3. **Parent-Child Granular Chunking (`src/ingestion/`)**: Subdivides documents into overlapping child chunks to optimize precise vector search alignment, but retrieves and passes their holistic parent chunks to the LLM to preserve complete surrounding context.
4. **Query Translation & HyDE (`src/transformation/`)**: Generates multi-query synonyms and Hypothetical Document Embeddings (HyDE) responses using Gemini models to bridge vocabulary gaps in asymmetric queries.
5. **Cross-Encoder Re-Ranking & Context Compression (`src/ranking/`)**: Performs token-to-token cross-attention scoring to re-evaluate doc relevance, stripping out irrelevant boilerplate down to individual sentence-level payloads.
6. **Corrective RAG / CRAG (`src/agents/crag_evaluator.py`)**: Grader agent evaluates retrieval relevance and dynamically triggers web search lookup (Tavily/Google Search) on irrelevant or ambiguous inputs.
7. **Self-RAG Reflection Loops (`src/agents/self_rag_evaluator.py`)**: Checks for hallucinations (Faithfulness check) against database contexts and verifies response usefulness (Utility check), looping back to generate again if checks fail.
8. **Structured Text-to-SQL DB Agent (`src/agents/text2sql.py`)**: Seamlessly reflects SQLite DDL schemas, translates natural analytical questions into valid SQLite SELECT commands, executes them, and formats tabular reports.
9. **NumPy Semantic Cache (`src/cache/`)**: Performs cosine similarity checks over past cached queries, serving identical semantic prompts in sub-milliseconds and bypassing LLM API charges.
10. **Enterprise Security Guardrails (`src/guardrails/`)**: Scans input prompts for injections, overrides, or jailbreaks, and sanitizes output vectors for secret token leaks (AWS/API keys) or PII (SSN/emails).

---

## 📂 Project Directory Structure

```
enterprise-advanced-rag/
│
├── .env.example                     # Environment variable template
├── .gitignore                       # Staging exclusions rules
├── requirements.txt                 # Exact python dependencies
├── setup.py                         # Local editable package installer
├── app.py                           # Beautiful Streamlit visual dashboard UI
│
├── data/                            # Raw corpus datasets & relational databases
│   ├── sample_corpus/               # Seeded company policy and technical guides
│   │   ├── hr_policy.md
│   │   ├── architecture_guide.md
│   │   └── cloud_security.md
│   └── business_metrics.db          # Seeded SQLite database (200 sales transactions)
│
├── notebooks/                       # 8 Step-by-Step Educational Jupyter Notebooks
│   ├── 01_data_ingestion_hybrid_search.ipynb
│   ├── 02_query_transformation_hyde.ipynb
│   ├── 03_reranking_context_compression.ipynb
│   ├── 04_corrective_rag_crag.ipynb
│   ├── 05_self_rag.ipynb
│   ├── 06_structured_text2sql.ipynb
│   ├── 07_caching_and_guardrails.ipynb
│   └── 08_langgraph_orchestrator.ipynb
│
├── scripts/                         # Seeding compilers and validation runners
│   ├── seed_db.py                   # Creates and seeds the mock SQLite transactional DB
│   ├── generate_notebooks.py        # Programmatically builds the Jupyter Notebook files
│   ├── test_integration.py          # Zero-dependency integration unit test runner
│   └── demo_run.py                  # Live scenario pipeline query demonstrations
│
├── src/                             # Production-grade Modular Codebase
│   ├── __init__.py                  # Exposes package API entrypoints
│   ├── config.py                    # Environment loader with MockLLM fallback engines
│   ├── utils/
│   │   ├── __init__.py
│   │   └── embedding_loader.py      # SentenceTransformers caching or MockEmbeddingModel
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── chunking.py              # Parent-Child splitting algorithms
│   ├── retrieval/
│   │   ├── __init__.py
│   │   └── hybrid.py                # NumPy Local Vector DB + BM25 Okapi + RRF Fusion
│   ├── transformation/
│   │   ├── __init__.py
│   │   └── hyde.py                  # Synonym rewrites and HyDE document generators
│   ├── ranking/
│   │   ├── __init__.py
│   │   └── reranker.py              # Cross-Encoder re-ranker + sentence compressors
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── crag_evaluator.py        # Correct/Incorrect Document Grader
│   │   ├── self_rag_evaluator.py    # Faithfulness and Utility verification nodes
│   │   └── text2sql.py              # DB schema reflection and Text-to-SQL queries
│   ├── cache/
│   │   ├── __init__.py
│   │   └── semantic_cache.py        # SQLite Cosine similarity cache store
│   └── guardrails/
│       ├── __init__.py
│       └── validator.py             # prompt injection blocking and PII redactor
│
└── tests/                           # Unit tests suite (pytest compatible structure)
    ├── __init__.py
    ├── test_guardrails.py
    ├── test_langgraph.py
    └── test_retrieval.py
```

---

## 🛠️ Installation & Setup

### 1. Initialize and configure the Environment
Create your `.env` configuration file from the template:
```bash
cp .env.example .env
```
*(Optional: Open `.env` and add your `GOOGLE_API_KEY` to hook up live Gemini generative engines instead of our Mock LLM Client fallback).*

### 2. Install Project Dependencies
To install the complete cognitive pipeline, run:
```bash
pip install -r requirements.txt
# Install the library locally in editable mode
pip install -e .
```

### 3. Run the Zero-Dependency Test Suite
To verify the vector similarity math, cache hits, guardrails, and graph state transitions are 100% green and operational immediately:
```bash
python scripts/test_integration.py
```

---

## 🚀 Interactive UI Dashboard

To launch our gorgeous Streamlit Command Dashboard which renders live pipeline latency metrics, active execution graph traces, SQL query records, and doc grader statuses directly in your browser, run:
```bash
streamlit run app.py
```

---

## 🧪 Production Usage Example

Using the Compiled LangGraph platform in your production Python scripts is incredibly simple:

```python
from src.graph.pipeline import AdvancedRAGPipeline
import os

# Initialize pipeline pointing to corpus directory
corpus_dir = "data/sample_corpus"
pipeline = AdvancedRAGPipeline(corpus_dir)

# 1. Execute unstructured corporate policy query (RAG path)
res_rag = pipeline.run("What are remote work home setup stipends?")
print("Answer:", res_rag["generation"])
print("Execution Path Trace:", res_rag["execution_trace"])

# 2. Execute structured analytics query (Text-to-SQL path)
res_sql = pipeline.run("What is the top SaaS product by sales revenue?")
print("Answer:", res_sql["generation"])
print("SQL Run:", res_sql.get("sql_query"))
```

---

## 🛡️ Zero-Dependency Resilience Design

To ensure the repository operates cleanly **immediately upon clone**, we programmatically compiled a zero-dependency fallback framework. If heavy NLP libraries or API keys are missing:
*   **Vector DB Fallback**: Replaced by a high-fidelity NumPy vector array executing dot-product Cosine Similarity.
*   **Embedding Model Fallback**: Replaced by `MockEmbeddingModel` yielding 384-dimensional normalized vectors with semantic term weight activations so semantic tests pass flawlessly.
*   **BM25 Fallback**: Replaced by a custom, NumPy-optimized `BM25Okapi` term statistics implementation.
*   **State Graph Fallback**: Replaced by a native Python state transitions manager executing state nodes and conditional edges identically to LangGraph.
