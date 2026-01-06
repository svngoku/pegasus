"""Main Pegasus RAG engine orchestrator."""

import hashlib
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from .config import PegasusConfig
from .models import PegasusDoc, SearchResult
from .chunking import chunk_text
from .embeddings import EmbeddingProvider
from .index import VectorIndexManager
from .storage import MetadataStore
from .search import SearchEngine


class Pegasus:
    """High-performance RAG engine combining vector search, FTS, and hybrid modes."""
    
    def __init__(self, config: PegasusConfig, openai_api_key: Optional[str] = None):
        self.config = config
        self._lock = threading.RLock()
        
        # Initialize components
        self.embedder = EmbeddingProvider(config.embedding_model, openai_api_key)
        self.index_manager = VectorIndexManager(
            config.index_path,
            config.embedding_dim,
            metric=config.metric,
            dtype=config.dtype,
            connectivity=config.connectivity,
            expansion_add=config.expansion_add,
            expansion_search=config.expansion_search,
        )
        self.metadata_store = MetadataStore(config.db_path)
        self.search_engine = SearchEngine(
            self.embedder,
            self.index_manager,
            self.metadata_store,
            config.hybrid_alpha,
        )
    
    def ingest(
        self,
        docs: Union[List[PegasusDoc], List[str]],
        corpus: str = "default",
        show_progress: bool = True,
        on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        use_tqdm: bool = False,
    ) -> Dict[str, int]:
        """
        Ingest documents into the RAG engine.
        
        Args:
            docs: List of PegasusDoc objects or raw text strings
            corpus: Corpus identifier for grouping chunks
            show_progress: Whether to print progress summary at end
            on_progress: Optional callback called after each document with stats:
                         {"doc_index": int, "total_docs": int, "chunks": int, 
                          "skipped": int, "current_doc_id": str}
            use_tqdm: Use tqdm progress bar (requires: pip install tqdm)
        
        Returns:
            Stats dict with 'chunks' and 'skipped' counts
        """
        # Normalize input
        if docs and isinstance(docs[0], str):
            docs = [PegasusDoc(text=doc) for doc in docs]
        
        stats = {"chunks": 0, "skipped": 0, "docs": len(docs)}
        
        # Setup progress bar if requested
        doc_iterator = docs
        pbar = None
        if use_tqdm:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(docs), desc="Ingesting", unit="doc")
                doc_iterator = docs  # Don't wrap, we'll update manually
            except ImportError:
                pass  # tqdm not installed, continue without
        
        with self._lock:
            for doc_idx, doc in enumerate(doc_iterator):
                # Chunk document
                chunks = chunk_text(
                    doc.text,
                    max_chars=self.config.chunk_size * 4,
                    overlap_chars=self.config.chunk_overlap * 4,
                    strategy=self.config.chunk_strategy,
                )
                
                # Generate embeddings
                embeddings = self.embedder.embed(chunks)
                
                # Ingest chunks
                for chunk_idx, (chunk_text_content, embedding) in enumerate(zip(chunks, embeddings)):
                    # Prepare chunk record
                    content_hash = hashlib.sha256(chunk_text_content.encode()).hexdigest()
                    
                    chunk_record = {
                        "corpus": corpus,
                        "doc_id": doc.doc_id,
                        "chunk_index": chunk_idx,
                        "content": chunk_text_content,
                        "content_hash": content_hash,
                        "source": doc.metadata.get("source", ""),
                        "title": doc.metadata.get("title", ""),
                        "page": doc.metadata.get("page"),
                        "metadata_json": json.dumps(doc.metadata) if doc.metadata else None,
                    }
                    
                    # Insert and index
                    chunk_id = self.metadata_store.insert_chunk(chunk_record)
                    
                    if chunk_id:
                        self.index_manager.add(chunk_id, embedding)
                        stats["chunks"] += 1
                    else:
                        stats["skipped"] += 1
                
                # Call progress callback
                if on_progress:
                    on_progress({
                        "doc_index": doc_idx + 1,
                        "total_docs": len(docs),
                        "chunks": stats["chunks"],
                        "skipped": stats["skipped"],
                        "current_doc_id": doc.doc_id,
                    })
                
                # Update tqdm
                if pbar:
                    pbar.update(1)
                    pbar.set_postfix(chunks=stats["chunks"], skipped=stats["skipped"])
        
        if pbar:
            pbar.close()
        
        self.index_manager.save()
        
        if show_progress:
            print(f"Ingested {stats['chunks']} chunks, skipped {stats['skipped']} duplicates")
        
        return stats
    
    def search(
        self,
        query: str,
        *,
        k: int = 10,
        corpus: Optional[str] = None,
        mode: str = "vector",
        hybrid_alpha: Optional[float] = None,
        filter_fn: Optional[Callable[[Dict], bool]] = None,
    ) -> List[SearchResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query text
            k: Number of results to return
            corpus: Filter by corpus
            mode: 'vector', 'keyword', or 'hybrid'
            hybrid_alpha: Weight for vector search in hybrid mode (0-1)
            filter_fn: Optional metadata filter function
        
        Returns:
            List of SearchResult objects
        """
        with self._lock:
            if mode == "keyword":
                return self.search_engine.keyword_search(query, k=k, corpus=corpus, filter_fn=filter_fn)
            elif mode == "hybrid":
                return self.search_engine.hybrid_search(
                    query, k=k, corpus=corpus, alpha=hybrid_alpha, filter_fn=filter_fn
                )
            else:  # vector
                return self.search_engine.vector_search(query, k=k, corpus=corpus, filter_fn=filter_fn)
    
    def delete_corpus(self, corpus: str) -> int:
        """Delete all chunks in a corpus."""
        with self._lock:
            return self.metadata_store.delete_corpus(corpus)
    
    def delete_chunk(self, chunk_id: int) -> bool:
        """
        Delete a single chunk by ID.
        
        Note: This removes from metadata but leaves the vector in the index.
        For full cleanup, rebuild the index or use delete_by_doc_id.
        
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            return self.metadata_store.delete_chunk(chunk_id)
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            doc_id: The document ID to delete
        
        Returns:
            Number of chunks deleted
        """
        with self._lock:
            return self.metadata_store.delete_by_doc_id(doc_id)
    
    def update_chunk(self, chunk_id: int, content: str) -> bool:
        """
        Update chunk content and re-embed.
        
        Args:
            chunk_id: The chunk ID to update
            content: New content
        
        Returns:
            True if updated, False if not found
        """
        with self._lock:
            # Update metadata
            if not self.metadata_store.update_chunk_content(chunk_id, content):
                return False
            
            # Re-embed and update index
            embedding = self.embedder.embed(content)[0]
            self.index_manager.add(chunk_id, embedding)
            self.index_manager.save()
            return True
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        with self._lock:
            return self.metadata_store.get_chunk(chunk_id)
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        with self._lock:
            return self.metadata_store.get_chunks_by_doc_id(doc_id)
    
    def list_corpora(self) -> List[Dict[str, Any]]:
        """List all corpora with statistics."""
        with self._lock:
            return self.metadata_store.list_corpora()
    
    def export_corpus(self, corpus: str, output_path: str) -> int:
        """
        Export a corpus to JSONL file for backup.
        
        Args:
            corpus: Corpus name to export
            output_path: Path to output JSONL file
        
        Returns:
            Number of chunks exported
        """
        from pathlib import Path
        
        with self._lock:
            cursor = self.metadata_store.conn.cursor()
            rows = cursor.execute(
                "SELECT * FROM chunks WHERE corpus = ?", (corpus,)
            ).fetchall()
            
            output_file = Path(output_path)
            count = 0
            
            with open(output_file, "w") as f:
                for row in rows:
                    chunk_data = dict(row)
                    # Remove auto-generated fields
                    chunk_data.pop("id", None)
                    chunk_data.pop("created_at", None)
                    chunk_data.pop("updated_at", None)
                    f.write(json.dumps(chunk_data) + "\n")
                    count += 1
            
            return count
    
    def import_corpus(
        self,
        input_path: str,
        corpus: Optional[str] = None,
        skip_duplicates: bool = True,
    ) -> Dict[str, int]:
        """
        Import a corpus from JSONL backup file.
        
        Args:
            input_path: Path to JSONL file
            corpus: Override corpus name (uses file's corpus if None)
            skip_duplicates: Skip chunks with existing content_hash
        
        Returns:
            Stats dict with 'imported' and 'skipped' counts
        """
        from pathlib import Path
        
        stats = {"imported": 0, "skipped": 0}
        
        with self._lock:
            with open(Path(input_path), "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    chunk_data = json.loads(line)
                    
                    # Override corpus if specified
                    if corpus:
                        chunk_data["corpus"] = corpus
                    
                    # Re-embed content
                    content = chunk_data.get("content", "")
                    if not content:
                        stats["skipped"] += 1
                        continue
                    
                    # Insert chunk
                    chunk_id = self.metadata_store.insert_chunk(chunk_data)
                    
                    if chunk_id:
                        # Generate and add embedding
                        embedding = self.embedder.embed(content)[0]
                        self.index_manager.add(chunk_id, embedding)
                        stats["imported"] += 1
                    else:
                        stats["skipped"] += 1
            
            self.index_manager.save()
        
        return stats
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            return {
                "index_size": len(self.index_manager),
                "db_path": self.config.db_path,
                "index_path": self.config.index_path,
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "dtype": self.config.dtype,
                "corpora": self.list_corpora(),
            }
    
    def save(self) -> None:
        """Persist index to disk."""
        with self._lock:
            self.index_manager.save()
    
    def close(self) -> None:
        """Close all connections and save."""
        with self._lock:
            self.save()
            self.metadata_store.close()


def create_pegasus(
    db_path: str = "pegasus.db",
    index_path: str = "pegasus.usearch",
    *,
    embedding_model: str = "text-embedding-3-large",
    dtype: str = "f16",
    openai_api_key: Optional[str] = None,
) -> Pegasus:
    """
    Create a Pegasus instance with sensible defaults.
    
    Example:
        >>> pegasus = create_pegasus("myrag.db", "myrag.usearch")
        >>> pegasus.ingest(load_sources(["./docs"]))
        >>> results = pegasus.search("How do I...?")
    """
    config = PegasusConfig(
        db_path=db_path,
        index_path=index_path,
        embedding_model=embedding_model,
        dtype=dtype,
    )
    return Pegasus(config, openai_api_key=openai_api_key)
