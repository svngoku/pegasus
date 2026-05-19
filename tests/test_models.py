from pegasus.models import PegasusDoc, SearchResult


def test_pegasus_doc_auto_id():
    """Test that doc_id is automatically and stably generated from content."""
    doc1 = PegasusDoc(text="Hello world")
    doc2 = PegasusDoc(text="Hello world")
    doc3 = PegasusDoc(text="Different content")

    assert doc1.doc_id is not None
    assert len(doc1.doc_id) == 16
    assert doc1.doc_id == doc2.doc_id
    assert doc1.doc_id != doc3.doc_id


def test_pegasus_doc_custom_id():
    """Test that custom doc_id is preserved."""
    doc = PegasusDoc(text="Hello world", doc_id="custom_123", metadata={"author": "test"})
    assert doc.doc_id == "custom_123"
    assert doc.metadata == {"author": "test"}


def test_search_result_structure():
    """Test SearchResult dataclass creation and values."""
    res = SearchResult(
        chunk_id=42,
        doc_id="doc_xyz",
        content="This is a chunk.",
        score=0.95,
        metadata={"source": "book"},
    )
    assert res.chunk_id == 42
    assert res.doc_id == "doc_xyz"
    assert res.content == "This is a chunk."
    assert res.score == 0.95
    assert res.metadata == {"source": "book"}
