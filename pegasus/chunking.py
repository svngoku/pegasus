"""Text chunking strategies for document processing."""

import re
from typing import List


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using regex."""
    # Handle common abbreviations and edge cases
    text = re.sub(r'([.!?])\s+', r'\1\n', text)
    sentences = [s.strip() for s in text.split('\n') if s.strip()]
    return sentences


def _split_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_text(
    text: str,
    *,
    max_chars: int = 2000,
    overlap_chars: int = 200,
    strategy: str = "sentence",
) -> List[str]:
    """
    Split text into overlapping chunks using various strategies.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap characters between chunks
        strategy: 'sentence', 'paragraph', or 'fixed'
    
    Returns:
        List of text chunks
    """
    if strategy == "sentence":
        units = _split_sentences(text)
    elif strategy == "paragraph":
        units = _split_paragraphs(text)
    else:  # fixed
        units = [text[i:i+max_chars] for i in range(0, len(text), max_chars - overlap_chars)]
        return units
    
    chunks = []
    current_chunk = ""
    
    for unit in units:
        if len(current_chunk) + len(unit) + 1 <= max_chars:
            current_chunk += (" " if current_chunk else "") + unit
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # Start new chunk with overlap
            overlap_idx = max(0, len(current_chunk) - overlap_chars)
            current_chunk = current_chunk[overlap_idx:] + " " + unit
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return [c.strip() for c in chunks if c.strip()]
