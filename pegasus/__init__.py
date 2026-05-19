# ruff: noqa: E402
# Set USER_AGENT early to suppress langchain warning
import os as _os

if not _os.environ.get("USER_AGENT"):
    _os.environ["USER_AGENT"] = "pegasus-rag/2.1.0"
del _os

"""
Pegasus v2 — High-Performance RAG Engine with USearch + SQLite

A production-ready RAG ingestor and retriever combining:
- USearch native HNSW index for blazing-fast vector search
- SQLite for metadata storage and FTS5 full-text search
- Hybrid search (vector + keyword) with RRF fusion

Key Features:
- Native HNSW index with SIMD acceleration (10x faster than FAISS)
- Half-precision (f16/bf16) support for 2x memory efficiency
- Memory-mapped index serving for large-scale deployments
- Hybrid search combining semantic + keyword matching
- Batch ingestion with automatic threading
- Deduplication and upsert support
- Multi-index support for billion-scale datasets
- Sentence-aware chunking
- Retry logic for embedding API calls
- Multiple embedding providers (OpenAI, HuggingFace, Jina AI)
- LRU cache for embedding queries
- LLM-based re-ranking
- REST API (FastAPI)
- Export/Import for data portability

References:
- USearch: https://github.com/unum-cloud/usearch
- HNSW Algorithm: https://arxiv.org/abs/1603.09320
"""

from .chunking import chunk_text
from .config import PegasusConfig
from .embeddings import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    HuggingFaceEmbedding,
    JinaEmbedding,
    create_embedding_provider,
    get_cache,
)
from .index import VectorIndexManager
from .integration import (
    EmbeddingConfig,
    PegasusClient,
    PegasusClientConfig,
    check_installation,
    create_client,
    quick_search,
)
from .loaders import load_sources
from .models import PegasusDoc, SearchResult
from .pegasus import Pegasus, create_pegasus
from .reranker import LLMReranker, rerank_results
from .search import SearchEngine
from .storage import MetadataStore

__version__ = "2.1.0"
__all__ = [
    # Core
    "PegasusConfig",
    "PegasusDoc",
    "SearchResult",
    "Pegasus",
    "create_pegasus",
    # Loaders & Chunking
    "load_sources",
    "chunk_text",
    # Embeddings
    "EmbeddingProvider",
    "BaseEmbeddingProvider",
    "HuggingFaceEmbedding",
    "JinaEmbedding",
    "create_embedding_provider",
    "get_cache",
    # Components
    "VectorIndexManager",
    "MetadataStore",
    "SearchEngine",
    # Re-ranking
    "LLMReranker",
    "rerank_results",
    # Integration (PyPI client)
    "PegasusClient",
    "PegasusClientConfig",
    "EmbeddingConfig",
    "create_client",
    "quick_search",
    "check_installation",
]
