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

from .config import PegasusConfig
from .models import PegasusDoc, SearchResult
from .loaders import load_sources
from .chunking import chunk_text
from .embeddings import (
    EmbeddingProvider,
    BaseEmbeddingProvider,
    HuggingFaceEmbedding,
    JinaEmbedding,
    create_embedding_provider,
    get_cache,
)
from .index import VectorIndexManager
from .storage import MetadataStore
from .search import SearchEngine
from .pegasus import Pegasus, create_pegasus
from .reranker import LLMReranker, rerank_results

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
]
