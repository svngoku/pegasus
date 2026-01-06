# Pegasus v2 — High-Performance RAG Engine

A production-ready Retrieval-Augmented Generation (RAG) engine combining **USearch's native HNSW index** for blazing-fast vector search with **SQLite** for metadata storage and hybrid search capabilities.

## Key Improvements Over v1

| Feature | Pegasus v1 | Pegasus v2 |
|---------|------------|------------|
| **Vector Search** | SQLite + USearch distance functions (brute-force) | Native HNSW index (logarithmic complexity) |
| **Search Speed** | O(n) per query | O(log n) per query |
| **Memory Efficiency** | f32 only | f16/bf16 support (2x savings) |
| **Search Modes** | Vector only | Vector, Keyword, Hybrid (RRF) |
| **Chunking** | Fixed character split | Sentence-aware splitting |
| **Deduplication** | None | Content-hash based |
| **Thread Safety** | Limited | Full RLock protection |
| **Retry Logic** | None | Exponential backoff |
| **Index Persistence** | Embedded in SQLite | Memory-mapped file serving |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Pegasus v2                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌───────────────┐    ┌──────────────────────────────────┐ │
│  │   Documents   │───▶│         Chunker                  │ │
│  │  (PDF, MD,    │    │  • Sentence-aware splitting      │ │
│  │   URL, TXT)   │    │  • Configurable overlap          │ │
│  └───────────────┘    └──────────────┬───────────────────┘ │
│                                      │                      │
│                                      ▼                      │
│                       ┌──────────────────────────────────┐ │
│                       │      Embedding Provider          │ │
│                       │  • OpenAI text-embedding-3-large │ │
│                       │  • Batched API calls             │ │
│                       │  • Retry with backoff            │ │
│                       └──────────────┬───────────────────┘ │
│                                      │                      │
│              ┌───────────────────────┴────────────────┐    │
│              ▼                                        ▼    │
│  ┌───────────────────────┐         ┌─────────────────────┐│
│  │   USearch HNSW Index  │         │       SQLite        ││
│  │  • Native C++ engine  │         │  • Chunk metadata   ││
│  │  • SIMD acceleration  │         │  • FTS5 full-text   ││
│  │  • f16/bf16 vectors   │         │  • Corpus filtering ││
│  │  • Memory-mapped I/O  │         │  • Deduplication    ││
│  └───────────────────────┘         └─────────────────────┘│
│              │                                        │    │
│              └────────────────┬───────────────────────┘    │
│                               ▼                            │
│                    ┌──────────────────────┐                │
│                    │    Search Engine     │                │
│                    │  • Vector search     │                │
│                    │  • Keyword search    │                │
│                    │  • Hybrid (RRF)      │                │
│                    └──────────────────────┘                │
└─────────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install usearch "openai>=1.0.0" langchain-community langchain-core \
    beautifulsoup4 pymupdf requests tenacity numpy
```

## Quick Start

```python
from pegasus_v2 import Pegasus, load_sources, create_pegasus

# Create engine with defaults
pegasus = create_pegasus("myrag.db", "myrag.usearch")

# Load documents from multiple sources
docs = load_sources([
    "./documents/",           # Directory of .md, .txt, .pdf files
    "https://example.com",    # Web pages
    "./specific_file.pdf",    # Single files
])

# Ingest with automatic chunking and embedding
stats = pegasus.ingest(docs, corpus="knowledge_base")
print(f"Indexed {stats['chunks']} chunks from {stats['docs']} documents")

# Search (multiple modes available)
results = pegasus.search("How do I configure authentication?", k=5)

for r in results:
    print(f"[{r.score:.3f}] {r.content[:200]}...")
    print(f"  Source: {r.metadata.get('source')}\n")

# Hybrid search combines semantic + keyword matching
results = pegasus.search(
    "authentication OAuth2 setup",
    mode="hybrid",
    hybrid_alpha=0.7,  # 70% vector, 30% keyword
)

# Always close when done
pegasus.close()
```

## Configuration

```python
from pegasus_v2 import Pegasus, PegasusConfig

config = PegasusConfig(
    # Embedding settings
    embedding_model="text-embedding-3-large",
    embedding_dim=3072,
    
    # USearch HNSW parameters
    metric="cos",           # 'cos', 'ip', 'l2sq'
    dtype="f16",            # 'f32', 'f16', 'bf16', 'i8'
    connectivity=32,        # M parameter (graph connectivity)
    expansion_add=128,      # efConstruction (index quality)
    expansion_search=64,    # ef (search quality)
    
    # Chunking
    chunk_size=512,         # tokens (~2000 chars)
    chunk_overlap=64,       # overlap tokens
    chunk_strategy="sentence",  # 'sentence', 'paragraph', 'fixed'
    
    # Search defaults
    default_k=10,
    hybrid_alpha=0.7,
    
    # Storage
    db_path="pegasus.db",
    index_path="pegasus.usearch",
)

pegasus = Pegasus(config)
```

## HNSW Parameter Tuning

| Parameter | Effect | Trade-off |
|-----------|--------|-----------|
| `connectivity` (M) | Graph edge density | Higher = better recall, more memory |
| `expansion_add` (efConstruction) | Index build thoroughness | Higher = better quality, slower indexing |
| `expansion_search` (ef) | Search beam width | Higher = better recall, slower search |

**Recommended settings:**

| Use Case | connectivity | expansion_add | expansion_search |
|----------|-------------|---------------|------------------|
| Low memory | 16 | 64 | 32 |
| **Balanced** | **32** | **128** | **64** |
| High recall | 64 | 256 | 128 |
| Maximum recall | 128 | 512 | 256 |

## Search Modes

### Vector Search (Default)
Pure semantic similarity using HNSW approximate nearest neighbors.

```python
results = pegasus.search("What is machine learning?", mode="vector")
```

### Keyword Search
Full-text search using SQLite FTS5 with BM25 ranking.

```python
results = pegasus.search("OAuth2 authentication", mode="keyword")
```

### Hybrid Search
Combines vector and keyword results using Reciprocal Rank Fusion (RRF).

```python
results = pegasus.search(
    "configure database connection pooling",
    mode="hybrid",
    hybrid_alpha=0.7,  # 70% vector weight, 30% keyword weight
)
```

## Memory Efficiency

Using `dtype="f16"` provides **2x memory savings** with minimal recall loss:

| dtype | Memory per 1M vectors (3072d) | Relative Recall |
|-------|-------------------------------|-----------------|
| f32 | ~12 GB | 100% |
| **f16** | **~6 GB** | ~99.5% |
| bf16 | ~6 GB | ~99.5% |
| i8 | ~3 GB | ~98% (cosine only) |

## Production Deployment

### Memory-Mapped Index Serving

For large indexes, use memory-mapped loading to avoid loading the entire index into RAM:

```python
# In your production code
pegasus = Pegasus(config)
pegasus._init_index(view_only=True)  # Memory-map instead of load
```

### Multi-Index for Billion-Scale

For extremely large datasets, partition into multiple indexes:

```python
from usearch.index import Indexes

# Create sharded indexes
indexes = [
    Pegasus(PegasusConfig(index_path=f"shard_{i}.usearch"))
    for i in range(num_shards)
]

# Or use USearch's native multi-index
from usearch.index import Indexes
multi = Indexes(paths=["shard_0.usearch", "shard_1.usearch", ...])
```

## API Reference

### `Pegasus`

**Methods:**
- `ingest(docs, corpus, ...)` — Ingest documents
- `search(query, k, mode, ...)` — Search for relevant chunks  
- `delete_corpus(corpus)` — Remove all chunks in a corpus
- `list_corpora()` — List all corpora with stats
- `get_stats()` — Get engine statistics
- `save()` — Persist index to disk
- `close()` — Close connections

### `load_sources(sources)`

Load documents from mixed sources (URLs, directories, files).

### `SearchResult`

Dataclass with fields:
- `chunk_id: int`
- `doc_id: str`
- `content: str`
- `score: float` (0-1, higher is better)
- `metadata: Dict[str, Any]`

## Benchmarks (Approximate)

| Operation | v1 (SQLite brute-force) | v2 (HNSW) | Speedup |
|-----------|------------------------|-----------|---------|
| Search 10k vectors | ~50ms | ~0.5ms | **100x** |
| Search 100k vectors | ~500ms | ~1ms | **500x** |
| Search 1M vectors | ~5s | ~2ms | **2500x** |
| Indexing 10k vectors | ~10s | ~2s | 5x |

*Note: Actual performance depends on hardware, embedding dimensions, and HNSW parameters.*

## License

MIT License — Use freely in your projects!
