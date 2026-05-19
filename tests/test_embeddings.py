import pytest

from pegasus.embeddings import EmbeddingCache, create_embedding_provider


def test_embedding_cache_operations():
    """Test basic EmbeddingCache operations (get, set, LRU eviction)."""
    cache = EmbeddingCache(maxsize=2)
    model = "test-model"

    # Set and get
    cache.set("text1", model, [0.1, 0.2])
    assert cache.get("text1", model) == [0.1, 0.2]

    # Miss
    assert cache.get("text2", model) is None

    # Stats check
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["size"] == 1

    # LRU Eviction
    cache.set("text2", model, [0.3, 0.4])
    cache.set("text3", model, [0.5, 0.6])  # Triggers eviction of text1 (oldest)

    assert cache.get("text1", model) is None
    assert cache.get("text2", model) == [0.3, 0.4]
    assert cache.get("text3", model) == [0.5, 0.6]

    # Clear
    cache.clear()
    assert cache.get("text2", model) is None
    assert cache.stats()["size"] == 0


def test_create_embedding_provider_invalid():
    """Test factory fails for unknown provider."""
    with pytest.raises(ValueError, match="Unknown provider"):
        create_embedding_provider("invalid-provider")
