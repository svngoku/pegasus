"""Search engine implementation (vector, keyword, hybrid)."""

import json
from typing import Any, Callable, Dict, List, Optional

from .embeddings import EmbeddingProvider
from .index import VectorIndexManager
from .models import SearchResult
from .storage import MetadataStore


class SearchEngine:
    """Handles vector, keyword, and hybrid search."""
    
    def __init__(
        self,
        embedder: EmbeddingProvider,
        index_manager: VectorIndexManager,
        metadata_store: MetadataStore,
        default_hybrid_alpha: float = 0.7,
    ):
        self.embedder = embedder
        self.index_manager = index_manager
        self.metadata_store = metadata_store
        self.default_hybrid_alpha = default_hybrid_alpha
    
    def vector_search(
        self,
        query: str,
        k: int = 10,
        corpus: Optional[str] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[SearchResult]:
        """Pure vector similarity search using HNSW."""
        # Generate query embedding
        query_embedding = self.embedder.embed(query)[0]
        
        # Search index
        fetch_k = k * 3 if (corpus or filter_fn) else k
        matches = self.index_manager.search(query_embedding, fetch_k)
        
        results = []
        for match in matches:
            chunk_id = int(match.key)
            distance = float(match.distance)
            
            # Fetch metadata
            chunk = self.metadata_store.get_chunk(chunk_id)
            if not chunk:
                continue
            
            # Apply filters
            if corpus and chunk["corpus"] != corpus:
                continue
            
            metadata = json.loads(chunk["metadata_json"]) if chunk["metadata_json"] else {}
            if filter_fn and not filter_fn(metadata):
                continue
            
            # Convert distance to similarity (0-1)
            similarity = max(0.0, 1.0 - distance)
            
            results.append(SearchResult(
                chunk_id=chunk_id,
                doc_id=chunk["doc_id"],
                content=chunk["content"],
                score=similarity,
                metadata={
                    "corpus": chunk["corpus"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "page": chunk["page"],
                    **metadata,
                },
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def keyword_search(
        self,
        query: str,
        k: int = 10,
        corpus: Optional[str] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[SearchResult]:
        """Full-text search using FTS5."""
        rows = self.metadata_store.search_fts(query, k=k*3, corpus=corpus)
        
        results = []
        for row in rows:
            metadata = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            
            if filter_fn and not filter_fn(metadata):
                continue
            
            # Normalize FTS5 rank to 0-1 score
            score = 1.0 / (1.0 - row["rank"])  # rank is negative
            
            results.append(SearchResult(
                chunk_id=row["id"],
                doc_id=row["doc_id"],
                content=row["content"],
                score=min(1.0, score),
                metadata={
                    "corpus": row["corpus"],
                    "source": row["source"],
                    "title": row["title"],
                    "page": row["page"],
                    **metadata,
                },
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def hybrid_search(
        self,
        query: str,
        k: int = 10,
        corpus: Optional[str] = None,
        alpha: Optional[float] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[SearchResult]:
        """Hybrid search using Reciprocal Rank Fusion (RRF)."""
        alpha = alpha or self.default_hybrid_alpha
        
        # Get results from both methods
        vector_results = self.vector_search(query, k=k*2, corpus=corpus, filter_fn=filter_fn)
        keyword_results = self.keyword_search(query, k=k*2, corpus=corpus, filter_fn=filter_fn)
        
        # Build rank maps
        vector_ranks = {r.chunk_id: i for i, r in enumerate(vector_results)}
        keyword_ranks = {r.chunk_id: i for i, r in enumerate(keyword_results)}
        
        # Combine all unique chunks
        all_chunks = {}
        for r in vector_results + keyword_results:
            if r.chunk_id not in all_chunks:
                all_chunks[r.chunk_id] = r
        
        # Calculate RRF scores
        rrf_constant = 60
        scored_results = []
        
        for chunk_id, result in all_chunks.items():
            vector_rank = vector_ranks.get(chunk_id, len(vector_results) + 1)
            keyword_rank = keyword_ranks.get(chunk_id, len(keyword_results) + 1)
            
            rrf_score = (
                alpha * (1.0 / (rrf_constant + vector_rank)) +
                (1 - alpha) * (1.0 / (rrf_constant + keyword_rank))
            )
            
            result.score = rrf_score
            scored_results.append(result)
        
        # Sort and return top k
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:k]
