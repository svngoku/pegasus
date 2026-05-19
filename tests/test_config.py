from pegasus.config import PegasusConfig


def test_pegasus_config_defaults():
    """Test default values of PegasusConfig."""
    config = PegasusConfig()
    assert config.embedding_model == "text-embedding-3-large"
    assert config.embedding_dim == 3072
    assert config.metric == "cos"
    assert config.dtype == "f16"
    assert config.connectivity == 32
    assert config.expansion_add == 128
    assert config.expansion_search == 64
    assert config.chunk_size == 512
    assert config.chunk_overlap == 64
    assert config.chunk_strategy == "sentence"
    assert config.default_k == 10
    assert config.hybrid_alpha == 0.7
    assert config.db_path == "pegasus.db"
    assert config.index_path == "pegasus.usearch"


def test_pegasus_config_custom():
    """Test custom values of PegasusConfig."""
    config = PegasusConfig(
        embedding_model="custom-model",
        embedding_dim=512,
        metric="l2sq",
        dtype="f32",
        connectivity=16,
        expansion_add=64,
        expansion_search=32,
        chunk_size=256,
        chunk_overlap=32,
        chunk_strategy="paragraph",
        default_k=5,
        hybrid_alpha=0.5,
        db_path="custom.db",
        index_path="custom.usearch",
    )
    assert config.embedding_model == "custom-model"
    assert config.embedding_dim == 512
    assert config.metric == "l2sq"
    assert config.dtype == "f32"
    assert config.connectivity == 16
    assert config.expansion_add == 64
    assert config.expansion_search == 32
    assert config.chunk_size == 256
    assert config.chunk_overlap == 32
    assert config.chunk_strategy == "paragraph"
    assert config.default_k == 5
    assert config.hybrid_alpha == 0.5
    assert config.db_path == "custom.db"
    assert config.index_path == "custom.usearch"
