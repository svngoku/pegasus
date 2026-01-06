"""
Integration module for using Pegasus from PyPI.

This module provides convenient factory functions and client classes
for integrating Pegasus into your applications, whether installed
locally or from PyPI.

Installation:
    pip install pegasus-rag
    # or
    uv add pegasus-rag

Basic Usage:
    from pegasus.integration import PegasusClient
    
    client = PegasusClient()
    client.ingest(["Hello world", "Machine learning is great"])
    results = client.search("AI")

With Custom Provider:
    from pegasus.integration import PegasusClient, EmbeddingConfig
    
    client = PegasusClient(
        embedding=EmbeddingConfig(provider="huggingface", model="all-MiniLM-L6-v2")
    )
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from .config import PegasusConfig
from .models import PegasusDoc, SearchResult
from .pegasus import Pegasus
from .embeddings import (
    BaseEmbeddingProvider,
    EmbeddingProvider,
    HuggingFaceEmbedding,
    JinaEmbedding,
    create_embedding_provider,
    get_cache,
)


# Type aliases
ProviderType = Literal["openai", "huggingface", "jina"]
SearchMode = Literal["vector", "keyword", "hybrid"]


@dataclass
class EmbeddingConfig:
    """Configuration for embedding providers."""
    
    provider: ProviderType = "openai"
    model: Optional[str] = None  # Uses provider default if None
    api_key: Optional[str] = None  # Uses env var if None
    
    # Provider-specific options
    hf_token: Optional[str] = None  # For HuggingFace private models
    jina_task: Optional[str] = None  # For Jina AI task optimization
    
    def create_provider(self) -> BaseEmbeddingProvider:
        """Create the configured embedding provider."""
        if self.provider == "openai":
            return EmbeddingProvider(
                model=self.model or "text-embedding-3-large",
                openai_api_key=self.api_key,
            )
        elif self.provider == "huggingface":
            return HuggingFaceEmbedding(
                model=self.model or "all-MiniLM-L6-v2",
                hf_token=self.hf_token,
            )
        elif self.provider == "jina":
            return JinaEmbedding(
                model=self.model or "jina-embeddings-v3",
                jina_api_key=self.api_key,
                task=self.jina_task,
            )
        else:
            raise ValueError(f"Unknown provider: {self.provider}")


@dataclass
class PegasusClientConfig:
    """Configuration for PegasusClient."""
    
    # Storage
    db_path: str = "pegasus.db"
    index_path: str = "pegasus.usearch"
    
    # Embedding
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    
    # Index settings
    dtype: str = "f16"
    connectivity: int = 32
    expansion_add: int = 128
    expansion_search: int = 64
    
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    chunk_strategy: str = "sentence"
    
    # Search defaults
    default_k: int = 10
    default_mode: SearchMode = "hybrid"
    hybrid_alpha: float = 0.7


class PegasusClient:
    """
    High-level client for Pegasus RAG engine.
    
    Provides a simplified interface for common RAG operations,
    suitable for integration into applications.
    
    Example:
        >>> client = PegasusClient()
        >>> client.ingest(["Doc 1 text", "Doc 2 text"], corpus="my_docs")
        >>> results = client.search("query", k=5)
        >>> for r in results:
        ...     print(f"{r.score:.2f}: {r.content[:50]}")
    """
    
    def __init__(
        self,
        config: Optional[PegasusClientConfig] = None,
        *,
        # Quick config overrides
        db_path: Optional[str] = None,
        index_path: Optional[str] = None,
        embedding: Optional[EmbeddingConfig] = None,
        provider: Optional[ProviderType] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize PegasusClient.
        
        Args:
            config: Full configuration object
            db_path: Database path (overrides config)
            index_path: Index path (overrides config)
            embedding: Embedding configuration (overrides config)
            provider: Quick provider selection (overrides embedding)
            model: Quick model selection (overrides embedding)
            api_key: Quick API key (overrides embedding)
        """
        # Build config
        self.config = config or PegasusClientConfig()
        
        if db_path:
            self.config.db_path = db_path
        if index_path:
            self.config.index_path = index_path
        if embedding:
            self.config.embedding = embedding
        if provider:
            self.config.embedding.provider = provider
        if model:
            self.config.embedding.model = model
        if api_key:
            self.config.embedding.api_key = api_key
        
        # Create embedding provider
        self._embedder = self.config.embedding.create_provider()
        
        # Create Pegasus config
        pegasus_config = PegasusConfig(
            db_path=self.config.db_path,
            index_path=self.config.index_path,
            embedding_model=self._embedder.model,
            embedding_dim=self._embedder.dimension,
            dtype=self.config.dtype,
            connectivity=self.config.connectivity,
            expansion_add=self.config.expansion_add,
            expansion_search=self.config.expansion_search,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunk_strategy=self.config.chunk_strategy,
            default_k=self.config.default_k,
            hybrid_alpha=self.config.hybrid_alpha,
        )
        
        # Create Pegasus instance
        self._pegasus = Pegasus(pegasus_config)
        
        # Replace embedder with our configured one
        self._pegasus.embedder = self._embedder
        self._pegasus.search_engine.embedder = self._embedder
    
    # ============================================================
    # Core Methods
    # ============================================================
    
    def ingest(
        self,
        texts: Union[List[str], List[PegasusDoc], List[Dict[str, Any]]],
        corpus: str = "default",
        **kwargs,
    ) -> Dict[str, int]:
        """
        Ingest documents into the RAG engine.
        
        Args:
            texts: List of texts, PegasusDoc objects, or dicts with 'text' key
            corpus: Corpus name for grouping documents
            **kwargs: Additional arguments for Pegasus.ingest()
        
        Returns:
            Stats dict with 'chunks', 'skipped', 'docs' counts
        
        Example:
            >>> client.ingest(["Hello", "World"], corpus="greetings")
            >>> client.ingest([{"text": "Doc", "metadata": {"source": "web"}}])
        """
        # Normalize input
        docs = []
        for item in texts:
            if isinstance(item, str):
                docs.append(PegasusDoc(text=item))
            elif isinstance(item, PegasusDoc):
                docs.append(item)
            elif isinstance(item, dict):
                docs.append(PegasusDoc(
                    text=item.get("text", ""),
                    metadata=item.get("metadata", {}),
                ))
            else:
                raise TypeError(f"Unsupported type: {type(item)}")
        
        return self._pegasus.ingest(docs, corpus=corpus, **kwargs)
    
    def search(
        self,
        query: str,
        k: Optional[int] = None,
        mode: Optional[SearchMode] = None,
        corpus: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query text
            k: Number of results (default: config.default_k)
            mode: Search mode: 'vector', 'keyword', 'hybrid' (default: config.default_mode)
            corpus: Filter by corpus name
            **kwargs: Additional arguments for Pegasus.search()
        
        Returns:
            List of SearchResult objects
        
        Example:
            >>> results = client.search("machine learning", k=5, mode="hybrid")
            >>> for r in results:
            ...     print(f"[{r.score:.2f}] {r.content[:50]}...")
        """
        return self._pegasus.search(
            query,
            k=k or self.config.default_k,
            mode=mode or self.config.default_mode,
            corpus=corpus,
            **kwargs,
        )
    
    def ask(
        self,
        query: str,
        k: int = 5,
        mode: SearchMode = "hybrid",
    ) -> List[Dict[str, Any]]:
        """
        Search and return results as dictionaries (JSON-friendly).
        
        Args:
            query: Search query
            k: Number of results
            mode: Search mode
        
        Returns:
            List of result dictionaries
        
        Example:
            >>> results = client.ask("What is RAG?")
            >>> print(results[0]["content"])
        """
        results = self.search(query, k=k, mode=mode)
        return [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "content": r.content,
                "score": r.score,
                "metadata": r.metadata,
            }
            for r in results
        ]
    
    # ============================================================
    # Management Methods
    # ============================================================
    
    def delete_corpus(self, corpus: str) -> int:
        """Delete all documents in a corpus."""
        return self._pegasus.delete_corpus(corpus)
    
    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for a document."""
        return self._pegasus.delete_by_doc_id(doc_id)
    
    def list_corpora(self) -> List[Dict[str, Any]]:
        """List all corpora with statistics."""
        return self._pegasus.list_corpora()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        stats = self._pegasus.get_stats()
        stats["cache"] = get_cache().stats()
        return stats
    
    def export(self, corpus: str, path: str) -> int:
        """Export a corpus to JSONL file."""
        return self._pegasus.export_corpus(corpus, path)
    
    def import_corpus(self, path: str, corpus: Optional[str] = None) -> Dict[str, int]:
        """Import a corpus from JSONL file."""
        return self._pegasus.import_corpus(path, corpus=corpus)
    
    # ============================================================
    # Context Manager
    # ============================================================
    
    def close(self) -> None:
        """Close the client and release resources."""
        self._pegasus.close()
    
    def __enter__(self) -> "PegasusClient":
        return self
    
    def __exit__(self, *args) -> None:
        self.close()


# ============================================================
# Convenience Functions
# ============================================================

def create_client(
    provider: ProviderType = "openai",
    model: Optional[str] = None,
    db_path: str = "pegasus.db",
    **kwargs,
) -> PegasusClient:
    """
    Quick factory function to create a PegasusClient.
    
    Args:
        provider: Embedding provider ('openai', 'huggingface', 'jina')
        model: Model name (uses provider default if not specified)
        db_path: Database path
        **kwargs: Additional PegasusClient arguments
    
    Returns:
        Configured PegasusClient
    
    Example:
        >>> # OpenAI (requires OPENAI_API_KEY)
        >>> client = create_client("openai")
        >>> 
        >>> # HuggingFace (local, free)
        >>> client = create_client("huggingface", model="all-MiniLM-L6-v2")
        >>> 
        >>> # Jina AI (requires JINA_API_KEY)
        >>> client = create_client("jina", model="jina-embeddings-v3")
    """
    return PegasusClient(
        provider=provider,
        model=model,
        db_path=db_path,
        **kwargs,
    )


def quick_search(
    query: str,
    texts: List[str],
    k: int = 5,
    provider: ProviderType = "openai",
) -> List[Dict[str, Any]]:
    """
    One-shot search: ingest texts and search immediately.
    
    Useful for quick experiments or one-time searches.
    Creates a temporary database that is cleaned up after.
    
    Args:
        query: Search query
        texts: List of texts to search through
        k: Number of results
        provider: Embedding provider
    
    Returns:
        List of result dictionaries
    
    Example:
        >>> texts = ["Python is great", "Java is verbose", "Rust is fast"]
        >>> results = quick_search("fast programming", texts, k=2)
        >>> print(results[0]["content"])
    """
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "temp.db"
        index_path = Path(tmpdir) / "temp.usearch"
        
        with create_client(
            provider=provider,
            db_path=str(db_path),
            index_path=str(index_path),
        ) as client:
            client.ingest(texts, show_progress=False)
            return client.ask(query, k=k)


# ============================================================
# Version check utility
# ============================================================

def check_installation() -> Dict[str, Any]:
    """
    Check Pegasus installation and available providers.
    
    Returns:
        Dictionary with version info and provider availability
    
    Example:
        >>> info = check_installation()
        >>> print(f"Version: {info['version']}")
        >>> print(f"Providers: {info['providers']}")
    """
    from . import __version__
    
    result = {
        "version": __version__,
        "providers": {},
        "env_vars": {},
    }
    
    # Check OpenAI
    result["env_vars"]["OPENAI_API_KEY"] = bool(os.environ.get("OPENAI_API_KEY"))
    result["providers"]["openai"] = result["env_vars"]["OPENAI_API_KEY"]
    
    # Check HuggingFace
    try:
        import sentence_transformers
        result["providers"]["huggingface"] = True
    except ImportError:
        result["providers"]["huggingface"] = False
    
    result["env_vars"]["HF_TOKEN"] = bool(os.environ.get("HF_TOKEN"))
    
    # Check Jina
    result["env_vars"]["JINA_API_KEY"] = bool(os.environ.get("JINA_API_KEY"))
    result["providers"]["jina"] = result["env_vars"]["JINA_API_KEY"]
    
    return result
