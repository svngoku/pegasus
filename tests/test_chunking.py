from pegasus.chunking import chunk_text


def test_chunk_text_fixed():
    """Test fixed-size chunking with overlaps."""
    text = "abcdefghijkl"
    # Chunk size 4, overlap 2
    # Chunks should be:
    # 0-4: abcd
    # 2-6: cdef
    # 4-8: efgh
    # 6-10: ghij
    # 8-12: ijkl
    chunks = chunk_text(text, max_chars=4, overlap_chars=2, strategy="fixed")
    assert chunks == ["abcd", "cdef", "efgh", "ghij", "ijkl", "kl"]


def test_chunk_text_sentence():
    """Test sentence-based chunking."""
    text = "This is sentence one. This is sentence two! And three?"
    chunks = chunk_text(text, max_chars=25, overlap_chars=5, strategy="sentence")
    # Each sentence fits individually
    assert len(chunks) >= 2
    assert "This is sentence one." in chunks[0]


def test_chunk_text_paragraph():
    """Test paragraph-based chunking."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, max_chars=40, overlap_chars=10, strategy="paragraph")
    assert chunks[0] == "Paragraph one. Paragraph two."
    assert chunks[1] == "graph two. Paragraph three."
