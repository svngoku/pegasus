"""Data models for Pegasus RAG engine."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class PegasusDoc:
    """A document with text content and metadata."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: Optional[str] = None
    
    def __post_init__(self):
        if self.doc_id is None:
            # Generate stable doc_id from content hash
            self.doc_id = hashlib.sha256(self.text.encode()).hexdigest()[:16]


@dataclass
class SearchResult:
    """A single search result."""
    chunk_id: int
    doc_id: str
    content: str
    score: float  # Lower is better for distance, normalized to similarity
    metadata: Dict[str, Any] = field(default_factory=dict)
