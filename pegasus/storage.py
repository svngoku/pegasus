"""SQLite metadata storage and full-text search."""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


# FTS5 special characters that need escaping
FTS5_SPECIAL_CHARS = set('"(){}[]+-*^~:\'?&|!')


def escape_fts5_query(query: str) -> str:
    """Escape a query string for safe FTS5 MATCH usage.
    
    FTS5 has special syntax characters that cause syntax errors when used
    in queries. This function escapes them by wrapping each token in double
    quotes and escaping any internal double quotes.
    """
    if not query or not query.strip():
        return '""'
    
    # Check if query contains any special characters
    has_special = any(c in FTS5_SPECIAL_CHARS for c in query)
    
    if not has_special:
        # Simple query without special chars - return as-is
        return query
    
    # Escape by wrapping in double quotes
    # Double any existing double quotes inside
    escaped = query.replace('"', '""')
    return f'"{escaped}"'


class MetadataStore:
    """Manages SQLite metadata storage and FTS5 full-text search."""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        cursor = self.conn.cursor()
        
        # Chunks table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY,
                corpus TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                content_hash TEXT UNIQUE,
                source TEXT,
                title TEXT,
                page INTEGER,
                metadata_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_corpus ON chunks(corpus)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_content_hash ON chunks(content_hash)")
        
        # FTS5 virtual table for full-text search
        cursor.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                content,
                content='chunks',
                content_rowid='id'
            )
        """)
        
        self.conn.commit()
    
    def insert_chunk(self, chunk: Dict[str, Any]) -> Optional[int]:
        """Insert a chunk. Returns row id if inserted, None if duplicate."""
        cursor = self.conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO chunks (
                    corpus, doc_id, chunk_index, content, content_hash,
                    source, title, page, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk["corpus"],
                chunk["doc_id"],
                chunk["chunk_index"],
                chunk["content"],
                chunk["content_hash"],
                chunk["source"],
                chunk["title"],
                chunk["page"],
                chunk["metadata_json"],
            ))
            
            chunk_id = cursor.lastrowid
            
            # Add to FTS5 index
            cursor.execute(
                "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
                (chunk_id, chunk["content"])
            )
            
            self.conn.commit()
            return chunk_id
        except sqlite3.IntegrityError:
            return None
    
    def get_chunk(self, chunk_id: int) -> Optional[Dict[str, Any]]:
        """Get a chunk by ID."""
        cursor = self.conn.cursor()
        row = cursor.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return dict(row) if row else None
    
    def search_fts(self, query: str, k: int = 10, corpus: Optional[str] = None) -> List[Dict[str, Any]]:
        """Full-text search using FTS5."""
        # Escape special FTS5 characters to prevent syntax errors
        safe_query = escape_fts5_query(query)
        
        sql = """
            SELECT c.*, fts.rank
            FROM chunks c
            JOIN chunks_fts fts ON c.id = fts.rowid
            WHERE chunks_fts MATCH ?
        """
        params = [safe_query]
        
        if corpus:
            sql += " AND c.corpus = ?"
            params.append(corpus)
        
        sql += " ORDER BY fts.rank LIMIT ?"
        params.append(k)
        
        cursor = self.conn.cursor()
        rows = cursor.execute(sql, params).fetchall()
        return [dict(row) for row in rows]
    
    def delete_corpus(self, corpus: str) -> int:
        """Delete all chunks in a corpus."""
        cursor = self.conn.cursor()
        
        # Get chunk IDs
        chunk_ids = cursor.execute(
            "SELECT id FROM chunks WHERE corpus = ?", (corpus,)
        ).fetchall()
        
        # Delete from FTS5
        for (chunk_id,) in chunk_ids:
            cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        
        # Delete from chunks
        cursor.execute("DELETE FROM chunks WHERE corpus = ?", (corpus,))
        self.conn.commit()
        
        return len(chunk_ids)
    
    def list_corpora(self) -> List[Dict[str, Any]]:
        """List all corpora with statistics."""
        cursor = self.conn.cursor()
        rows = cursor.execute("""
            SELECT corpus, COUNT(*) as chunks, COUNT(DISTINCT doc_id) as docs
            FROM chunks
            GROUP BY corpus
        """).fetchall()
        return [dict(row) for row in rows]
    
    def delete_chunk(self, chunk_id: int) -> bool:
        """Delete a single chunk by ID. Returns True if deleted."""
        cursor = self.conn.cursor()
        
        # Check if exists
        exists = cursor.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not exists:
            return False
        
        # Delete from FTS5
        cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        # Delete from chunks
        cursor.execute("DELETE FROM chunks WHERE id = ?", (chunk_id,))
        self.conn.commit()
        return True
    
    def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks for a document. Returns count deleted."""
        cursor = self.conn.cursor()
        
        # Get chunk IDs
        chunk_ids = cursor.execute(
            "SELECT id FROM chunks WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        
        # Delete from FTS5
        for (cid,) in chunk_ids:
            cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (cid,))
        
        # Delete from chunks
        cursor.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.commit()
        return len(chunk_ids)
    
    def update_chunk_content(self, chunk_id: int, content: str) -> bool:
        """Update chunk content. Returns True if updated."""
        cursor = self.conn.cursor()
        
        # Check if exists
        exists = cursor.execute("SELECT 1 FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        if not exists:
            return False
        
        import hashlib
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        # Update content and hash
        cursor.execute("""
            UPDATE chunks 
            SET content = ?, content_hash = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        """, (content, content_hash, chunk_id))
        
        # Update FTS5
        cursor.execute("DELETE FROM chunks_fts WHERE rowid = ?", (chunk_id,))
        cursor.execute(
            "INSERT INTO chunks_fts(rowid, content) VALUES (?, ?)",
            (chunk_id, content)
        )
        
        self.conn.commit()
        return True
    
    def get_chunks_by_doc_id(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        cursor = self.conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,)
        ).fetchall()
        return [dict(row) for row in rows]
    
    def close(self) -> None:
        """Close database connection."""
        self.conn.close()
