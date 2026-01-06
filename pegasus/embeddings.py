"""Embedding generation with caching and multiple provider support."""

import hashlib
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Union

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


class EmbeddingCache:
    """LRU cache for embeddings to avoid redundant API calls."""
    
    def __init__(self, maxsize: int = 1000):
        self._cache: Dict[str, List[float]] = {}
        self._maxsize = maxsize
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0
    
    def _hash_text(self, text: str, model: str) -> str:
        """Create a hash key for text + model combination."""
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:16]
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get cached embedding if exists."""
        key = self._hash_text(text, model)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        self._misses += 1
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Cache an embedding."""
        key = self._hash_text(text, model)
        if key not in self._cache:
            # Evict oldest if at capacity
            if len(self._cache) >= self._maxsize:
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            self._access_order.append(key)
        self._cache[key] = embedding
    
    def stats(self) -> Dict[str, int]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "size": len(self._cache),
            "maxsize": self._maxsize,
        }
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0


# Global cache instance
_embedding_cache = EmbeddingCache(maxsize=1000)


def get_cache() -> EmbeddingCache:
    """Get the global embedding cache."""
    return _embedding_cache


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    def __init__(self, model: str, use_cache: bool = True):
        self.model = model
        self.use_cache = use_cache
    
    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts (provider-specific)."""
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return embedding dimension for this model."""
        pass
    
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Generate embeddings with caching.
        
        Args:
            texts: Single text or list of texts
        
        Returns:
            List of embedding vectors
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.use_cache:
            return self._embed_batch(texts)
        
        cache = get_cache()
        results: List[Optional[List[float]]] = [None] * len(texts)
        texts_to_embed: List[Tuple[int, str]] = []
        
        # Check cache first
        for i, text in enumerate(texts):
            cached = cache.get(text, self.model)
            if cached is not None:
                results[i] = cached
            else:
                texts_to_embed.append((i, text))
        
        # Embed uncached texts
        if texts_to_embed:
            indices, uncached_texts = zip(*texts_to_embed)
            new_embeddings = self._embed_batch(list(uncached_texts))
            
            for idx, text, embedding in zip(indices, uncached_texts, new_embeddings):
                cache.set(text, self.model, embedding)
                results[idx] = embedding
        
        return results  # type: ignore


class EmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider with retries and caching."""
    
    # Known dimensions for OpenAI models
    MODEL_DIMENSIONS = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536,
    }
    
    def __init__(self, model: str, openai_api_key: Optional[str] = None, use_cache: bool = True):
        super().__init__(model, use_cache)
        self.api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass openai_api_key parameter."
            )
        self.client = OpenAI(api_key=self.api_key)
    
    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model, 3072)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via OpenAI API."""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
        )
        
        # Sort by index to ensure correct order
        embeddings = sorted(response.data, key=lambda x: x.index)
        return [emb.embedding for emb in embeddings]


# Alias for backwards compatibility
OpenAIEmbedding = EmbeddingProvider


class HuggingFaceEmbedding(BaseEmbeddingProvider):
    """
    HuggingFace sentence-transformers embedding provider (local, free).
    
    Requires: uv add sentence-transformers
    
    Note: HuggingFace token is optional but recommended for private models
    and to avoid rate limits. Set HF_TOKEN environment variable.
    
    Example:
        >>> embedder = HuggingFaceEmbedding("all-MiniLM-L6-v2")
        >>> embeddings = embedder.embed(["Hello world"])
    """
    
    # Known dimensions for common models
    MODEL_DIMENSIONS = {
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "all-mpnet-base-v2": 768,
        "multi-qa-mpnet-base-dot-v1": 768,
        "all-distilroberta-v1": 768,
        "paraphrase-MiniLM-L6-v2": 384,
        "BAAI/bge-small-en-v1.5": 384,
        "BAAI/bge-base-en-v1.5": 768,
        "BAAI/bge-large-en-v1.5": 1024,
    }
    
    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        hf_token: Optional[str] = None,
        use_cache: bool = True,
    ):
        """
        Initialize HuggingFace embedding provider.
        
        Args:
            model: Model name from HuggingFace Hub
            hf_token: Optional HuggingFace token for private models (or set HF_TOKEN env var)
            use_cache: Enable embedding cache
        """
        super().__init__(model, use_cache)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self._model = None
        self._dimension: Optional[int] = None
    
    def _load_model(self):
        """Lazy-load the model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Run: uv add sentence-transformers"
                )
            # Pass token if available (for private models)
            self._model = SentenceTransformer(
                self.model,
                token=self.hf_token,
            )
            # Get actual dimension from model
            self._dimension = self._model.get_sentence_embedding_dimension()
        return self._model
    
    @property
    def dimension(self) -> int:
        if self._dimension is not None:
            return self._dimension
        # Try known dimensions first
        if self.model in self.MODEL_DIMENSIONS:
            return self.MODEL_DIMENSIONS[self.model]
        # Load model to get dimension
        self._load_model()
        return self._dimension or 384
    
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using sentence-transformers."""
        model = self._load_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class JinaEmbedding(BaseEmbeddingProvider):
    """
    Jina AI embedding provider (API-based).
    
    Requires: JINA_API_KEY environment variable
    
    Example:
        >>> embedder = JinaEmbedding("jina-embeddings-v3")
        >>> embeddings = embedder.embed(["Hello world"])
    """
    
    API_URL = "https://api.jina.ai/v1/embeddings"
    
    # Known dimensions for Jina models
    MODEL_DIMENSIONS = {
        "jina-embeddings-v3": 1024,
        "jina-embeddings-v2-base-en": 768,
        "jina-embeddings-v2-small-en": 512,
        "jina-clip-v2": 1024,
        "jina-embeddings-v4": 2048,
    }
    
    def __init__(
        self,
        model: str = "jina-embeddings-v3",
        jina_api_key: Optional[str] = None,
        use_cache: bool = True,
        task: Optional[str] = None,
    ):
        """
        Initialize Jina embedding provider.
        
        Args:
            model: Jina model name
            jina_api_key: API key (or set JINA_API_KEY env var)
            use_cache: Enable caching
            task: Optional task type for optimization:
                  'retrieval.query', 'retrieval.passage', 'text-matching'
        """
        super().__init__(model, use_cache)
        self.api_key = jina_api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Jina API key required. Set JINA_API_KEY environment variable "
                "or pass jina_api_key parameter."
            )
        self.task = task
    
    @property
    def dimension(self) -> int:
        return self.MODEL_DIMENSIONS.get(self.model, 1024)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings via Jina AI API."""
        import requests
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        payload = {
            "model": self.model,
            "input": texts,
        }
        
        if self.task:
            payload["task"] = self.task
        
        response = requests.post(self.API_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        data = response.json()
        # Extract embeddings in order
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings


# ============ Provider Factory ============

def create_embedding_provider(
    provider: str = "openai",
    model: Optional[str] = None,
    **kwargs
) -> BaseEmbeddingProvider:
    """
    Factory function to create embedding providers.
    
    Args:
        provider: Provider name ('openai', 'huggingface', 'jina')
        model: Model name (uses provider default if not specified)
        **kwargs: Additional provider-specific arguments
    
    Returns:
        Configured embedding provider
    
    Example:
        >>> # OpenAI (default)
        >>> embedder = create_embedding_provider("openai", "text-embedding-3-small")
        >>> 
        >>> # HuggingFace (local, free)
        >>> embedder = create_embedding_provider("huggingface", "all-MiniLM-L6-v2")
        >>> 
        >>> # Jina AI
        >>> embedder = create_embedding_provider("jina", "jina-embeddings-v3")
    """
    provider = provider.lower()
    
    if provider in ("openai", "openai-embedding"):
        model = model or "text-embedding-3-large"
        return EmbeddingProvider(model, **kwargs)
    
    elif provider in ("huggingface", "hf", "sentence-transformers"):
        model = model or "all-MiniLM-L6-v2"
        return HuggingFaceEmbedding(model, **kwargs)
    
    elif provider in ("jina", "jina-ai"):
        model = model or "jina-embeddings-v3"
        return JinaEmbedding(model, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Supported: 'openai', 'huggingface', 'jina'"
        )
