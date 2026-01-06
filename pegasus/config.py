"""Configuration models for Pegasus RAG engine."""

from dataclasses import dataclass


@dataclass
class PegasusConfig:
    """Configuration for the Pegasus RAG engine."""
    
    # Embedding settings
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072  # text-embedding-3-large dimension
    
    # USearch HNSW parameters (tuned for quality/speed balance)
    metric: str = "cos"  # 'cos', 'ip', 'l2sq'
    dtype: str = "f16"   # 'f32', 'f16', 'bf16', 'i8' - f16 gives 2x memory savings
    connectivity: int = 32      # M parameter - higher = better recall, more memory
    expansion_add: int = 128    # efConstruction - higher = better index quality
    expansion_search: int = 64  # ef - higher = better search recall
    
    # Chunking settings
    chunk_size: int = 512       # tokens (roughly 4 chars per token)
    chunk_overlap: int = 64     # overlap tokens
    chunk_strategy: str = "sentence"  # 'sentence', 'paragraph', 'fixed'
    
    # Search settings
    default_k: int = 10
    hybrid_alpha: float = 0.7  # Weight for vector search in hybrid (0-1)
    
    # Storage paths
    db_path: str = "pegasus.db"
    index_path: str = "pegasus.usearch"
