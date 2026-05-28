from setuptools import setup, find_packages

setup(
    name="enterprise-advanced-rag",
    version="1.0.0",
    description="Enterprise Advanced RAG with Hybrid Search, ReRanking, HyDE, CRAG, Self-RAG, Text2SQL, Caching and Guardrails in LangGraph",
    author="Advanced Agentic Coding Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "langchain",
        "langchain-community",
        "langchain-chroma",
        "langgraph",
        "chromadb",
        "sentence-transformers",
        "rank-bm25",
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "python-dotenv",
        "pydantic",
        "tavily-python",
        "pytest"
    ],
)
