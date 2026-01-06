"""FastAPI REST API wrapper for Pegasus RAG engine."""

from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .pegasus import Pegasus, create_pegasus
from .models import PegasusDoc
from .embeddings import get_cache


# ============ Request/Response Models ============

class IngestRequest(BaseModel):
    """Request body for document ingestion."""
    texts: List[str] = Field(..., description="List of text documents to ingest")
    corpus: str = Field(default="default", description="Corpus name for grouping")
    metadata: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Optional metadata for each document"
    )


class IngestResponse(BaseModel):
    """Response from ingestion."""
    chunks: int
    skipped: int
    docs: int


class SearchRequest(BaseModel):
    """Request body for search."""
    query: str = Field(..., description="Search query")
    k: int = Field(default=10, ge=1, le=100, description="Number of results")
    corpus: Optional[str] = Field(default=None, description="Filter by corpus")
    mode: str = Field(
        default="hybrid",
        description="Search mode: 'vector', 'keyword', or 'hybrid'"
    )
    hybrid_alpha: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Vector weight for hybrid search (0-1)"
    )


class SearchResultItem(BaseModel):
    """Single search result."""
    chunk_id: int
    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from search."""
    results: List[SearchResultItem]
    query: str
    mode: str
    count: int


class CorpusInfo(BaseModel):
    """Corpus statistics."""
    corpus: str
    chunks: int
    docs: int


class StatsResponse(BaseModel):
    """Engine statistics."""
    index_size: int
    db_path: str
    index_path: str
    embedding_model: str
    embedding_dim: int
    dtype: str
    corpora: List[CorpusInfo]
    cache_stats: Dict[str, Any]


class DeleteResponse(BaseModel):
    """Response from delete operations."""
    deleted: int
    corpus: str


class ChunkUpdateRequest(BaseModel):
    """Request body for chunk update."""
    content: str = Field(..., description="New chunk content")


# ============ App Factory ============

def create_app(
    db_path: str = "pegasus.db",
    index_path: str = "pegasus.usearch",
    **kwargs
) -> FastAPI:
    """
    Create a FastAPI app wrapping a Pegasus instance.
    
    Args:
        db_path: Path to SQLite database
        index_path: Path to USearch index
        **kwargs: Additional arguments for create_pegasus
    
    Returns:
        FastAPI app instance
    """
    
    # Store pegasus instance
    pegasus_instance: Optional[Pegasus] = None
    
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal pegasus_instance
        pegasus_instance = create_pegasus(db_path, index_path, **kwargs)
        yield
        if pegasus_instance:
            pegasus_instance.close()
    
    app = FastAPI(
        title="Pegasus RAG API",
        description="High-performance RAG engine with vector, keyword, and hybrid search",
        version="2.0.0",
        lifespan=lifespan,
    )
    
    def get_pegasus() -> Pegasus:
        if pegasus_instance is None:
            raise HTTPException(status_code=503, detail="Pegasus not initialized")
        return pegasus_instance
    
    # ============ Endpoints ============
    
    @app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
    async def ingest_documents(request: IngestRequest):
        """
        Ingest documents into the RAG engine.
        
        Documents are chunked, embedded, and indexed for search.
        """
        pegasus = get_pegasus()
        
        # Build PegasusDoc objects
        docs = []
        for i, text in enumerate(request.texts):
            metadata = request.metadata[i] if request.metadata and i < len(request.metadata) else {}
            docs.append(PegasusDoc(text=text, metadata=metadata))
        
        stats = pegasus.ingest(docs, corpus=request.corpus, show_progress=False)
        return IngestResponse(**stats)
    
    @app.post("/search", response_model=SearchResponse, tags=["Search"])
    async def search(request: SearchRequest):
        """
        Search for relevant chunks.
        
        Supports three modes:
        - **vector**: Pure semantic similarity
        - **keyword**: Full-text search with BM25
        - **hybrid**: Combines both with RRF fusion (recommended)
        """
        pegasus = get_pegasus()
        
        if request.mode not in ("vector", "keyword", "hybrid"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be 'vector', 'keyword', or 'hybrid'"
            )
        
        results = pegasus.search(
            request.query,
            k=request.k,
            corpus=request.corpus,
            mode=request.mode,
            hybrid_alpha=request.hybrid_alpha,
        )
        
        return SearchResponse(
            results=[
                SearchResultItem(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    content=r.content,
                    score=r.score,
                    metadata=r.metadata,
                )
                for r in results
            ],
            query=request.query,
            mode=request.mode,
            count=len(results),
        )
    
    @app.get("/corpora", response_model=List[CorpusInfo], tags=["Management"])
    async def list_corpora():
        """List all corpora with statistics."""
        pegasus = get_pegasus()
        corpora = pegasus.list_corpora()
        return [CorpusInfo(**c) for c in corpora]
    
    @app.get("/stats", response_model=StatsResponse, tags=["Management"])
    async def get_stats():
        """Get engine statistics including cache info."""
        pegasus = get_pegasus()
        stats = pegasus.get_stats()
        stats["cache_stats"] = get_cache().stats()
        return StatsResponse(
            index_size=stats["index_size"],
            db_path=stats["db_path"],
            index_path=stats["index_path"],
            embedding_model=stats["embedding_model"],
            embedding_dim=stats["embedding_dim"],
            dtype=stats["dtype"],
            corpora=[CorpusInfo(**c) for c in stats["corpora"]],
            cache_stats=stats["cache_stats"],
        )
    
    @app.delete("/corpus/{corpus_name}", response_model=DeleteResponse, tags=["Management"])
    async def delete_corpus(corpus_name: str):
        """Delete all chunks in a corpus."""
        pegasus = get_pegasus()
        deleted = pegasus.delete_corpus(corpus_name)
        return DeleteResponse(deleted=deleted, corpus=corpus_name)
    
    @app.get("/chunk/{chunk_id}", tags=["Chunks"])
    async def get_chunk(chunk_id: int):
        """Get a chunk by ID."""
        pegasus = get_pegasus()
        chunk = pegasus.get_chunk(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        return chunk
    
    @app.delete("/chunk/{chunk_id}", tags=["Chunks"])
    async def delete_chunk(chunk_id: int):
        """Delete a chunk by ID."""
        pegasus = get_pegasus()
        if not pegasus.delete_chunk(chunk_id):
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        return {"deleted": True, "chunk_id": chunk_id}
    
    @app.put("/chunk/{chunk_id}", tags=["Chunks"])
    async def update_chunk(chunk_id: int, request: ChunkUpdateRequest):
        """Update a chunk's content (re-embeds automatically)."""
        pegasus = get_pegasus()
        if not pegasus.update_chunk(chunk_id, request.content):
            raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
        return {"updated": True, "chunk_id": chunk_id}
    
    @app.delete("/doc/{doc_id}", tags=["Documents"])
    async def delete_document(doc_id: str):
        """Delete all chunks for a document."""
        pegasus = get_pegasus()
        deleted = pegasus.delete_by_doc_id(doc_id)
        return {"deleted": deleted, "doc_id": doc_id}
    
    @app.post("/cache/clear", tags=["Cache"])
    async def clear_cache():
        """Clear the embedding cache."""
        cache = get_cache()
        stats_before = cache.stats()
        cache.clear()
        return {"cleared": True, "entries_cleared": stats_before["size"]}
    
    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "pegasus"}
    
    return app


# Default app for `uvicorn pegasus.api:app`
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
