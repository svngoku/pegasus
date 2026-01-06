# Pegasus Codebase Structure

## Overview

Pegasus is organized as a modular Python package with clear separation of concerns:

```
pegasus/
├── __init__.py              # Package exports
├── config.py                # Configuration dataclass
├── models.py                # Data models (PegasusDoc, SearchResult)
├── loaders.py               # Document loading (URLs, PDFs, TXT, MD)
├── chunking.py              # Text chunking strategies
├── embeddings.py            # OpenAI embedding provider
├── index.py                 # USearch HNSW vector index
├── storage.py               # SQLite metadata & FTS5
├── search.py                # Search engine (vector/keyword/hybrid)
├── pegasus.py               # Main orchestrator
└── cli.py                   # CLI demo
```

## Module Responsibilities

### `config.py` — Configuration
- `PegasusConfig`: Single dataclass for all engine settings
- Embedding model, HNSW parameters, chunking, search defaults
- Storage paths (SQLite, USearch index)

**Dependencies:** None

---

### `models.py` — Data Models
- `PegasusDoc`: Document with text and metadata
- `SearchResult`: Search result with chunk content and score
- Auto-generates stable `doc_id` from content hash

**Dependencies:** dataclasses, hashlib

---

### `loaders.py` — Document Loading
- `load_sources()`: Multi-source loader supporting:
  - URLs (HTML via BeautifulSoup)
  - Directories (recursively)
  - Single files (.txt, .md, .mdx, .pdf)
- Uses LangChain loaders under the hood

**Dependencies:** langchain_community, pathlib

---

### `chunking.py` — Text Chunking
- `chunk_text()`: Main function with multiple strategies
- `_split_sentences()`: Sentence-aware chunking
- `_split_paragraphs()`: Paragraph-aware chunking
- Fixed-size fallback for other content

**Dependencies:** re

---

### `embeddings.py` — Embedding Generation
- `EmbeddingProvider`: OpenAI API wrapper
  - Automatic retry with exponential backoff (tenacity)
  - Batch embedding support
  - API key from env or parameter

**Dependencies:** openai, tenacity

---

### `index.py` — Vector Index Manager
- `VectorIndexManager`: USearch HNSW wrapper
  - Initialize/restore index from disk
  - Add/search/remove operations
  - Metric selection (cosine, IP, L2)
  - Dtype support (f32, f16, bf16, i8)
  - Automatic HNSW parameter tuning

**Dependencies:** usearch, numpy

---

### `storage.py` — Metadata Storage
- `MetadataStore`: SQLite + FTS5 wrapper
  - **chunks** table: all chunk metadata
  - **chunks_fts**: Virtual FTS5 table for full-text search
  - Indexes on corpus, doc_id, content_hash
  - Deduplication via content_hash unique constraint
  - Methods: insert, get, search, delete, list_corpora

**Dependencies:** sqlite3, json

---

### `search.py` — Search Engine
- `SearchEngine`: Plugs together vector, keyword, and hybrid search
  - `vector_search()`: Pure HNSW similarity
  - `keyword_search()`: FTS5 full-text
  - `hybrid_search()`: RRF-based fusion (configurable alpha)
  - Optional corpus and metadata filtering

**Dependencies:** All other components

---

### `pegasus.py` — Main Engine
- `Pegasus`: Orchestrator combining all components
  - Thread-safe with RLock
  - `ingest()`: Chunk, embed, dedupe, index, store
  - `search()`: Dispatch to search engine
  - `delete_corpus()`, `list_corpora()`, `get_stats()`
  - `save()`, `close()`: Lifecycle management
- `create_pegasus()`: Convenience factory with defaults

**Dependencies:** All other modules

---

### `cli.py` — Demo & CLI
- `demo()`: Sample usage with hardcoded documents
- Can be extended for real CLI if needed

**Dependencies:** pegasus components

---

## Data Flow

```
User Documents
    ↓
load_sources() → PegasusDoc[]
    ↓
Pegasus.ingest()
    ├→ chunk_text() → chunks[]
    ├→ EmbeddingProvider.embed() → embeddings[]
    └→ for each (chunk, embedding):
        ├→ MetadataStore.insert_chunk() → chunk_id
        └→ VectorIndexManager.add(chunk_id, embedding)
    ↓
[Index saved, DB persisted]

Query
    ↓
Pegasus.search()
    ├→ EmbeddingProvider.embed() → query_embedding
    └→ SearchEngine dispatches:
        ├→ vector_search(): VectorIndexManager.search() + MetadataStore.get_chunk()
        ├→ keyword_search(): MetadataStore.search_fts()
        └→ hybrid_search(): RRF fusion of both
    ↓
SearchResult[]
```

## Design Principles

1. **Single Responsibility** — Each module has one job
2. **Composability** — Components are pluggable
3. **Immutability** — Configs are frozen dataclasses
4. **Thread Safety** — Pegasus uses RLock for concurrent access
5. **Error Handling** — Graceful fallbacks (retry, skip duplicates)
6. **Testability** — Each module can be tested independently

## Usage Examples

### Basic Setup
```python
from pegasus import create_pegasus, load_sources

pegasus = create_pegasus("db.db", "index.usearch")
docs = load_sources(["./documents/", "https://example.com"])
stats = pegasus.ingest(docs, corpus="knowledge")
results = pegasus.search("How do I...?", mode="hybrid")
pegasus.close()
```

### Component-Level Usage
```python
from pegasus.config import PegasusConfig
from pegasus.embeddings import EmbeddingProvider
from pegasus.index import VectorIndexManager
from pegasus.storage import MetadataStore
from pegasus.search import SearchEngine

# Build custom pipeline
config = PegasusConfig(dtype="f32", expansion_search=128)
embedder = EmbeddingProvider(config.embedding_model)
index = VectorIndexManager(config.index_path, config.embedding_dim, dtype=config.dtype)
store = MetadataStore(config.db_path)
engine = SearchEngine(embedder, index, store)

# Use directly
embedding = embedder.embed("query text")
index.add(1, embedding)
results = engine.vector_search("query text", k=10)
```

## Extension Points

### Custom Chunking Strategy
Extend `chunking.py` with new strategy in `chunk_text()`:
```python
elif strategy == "custom":
    # Your logic here
    chunks = my_chunking_logic(text)
```

### Custom Embedding Provider
Subclass or replace `EmbeddingProvider`:
```python
class HuggingFaceEmbedder(EmbeddingProvider):
    def embed(self, texts):
        # HuggingFace logic
```

### Custom Metadata Filters
Pass `filter_fn` to `search()`:
```python
results = pegasus.search(
    "query",
    filter_fn=lambda meta: meta.get("source") == "trusted"
)
```

### Custom Metric or HNSW Parameters
Use `PegasusConfig`:
```python
config = PegasusConfig(
    metric="l2sq",
    connectivity=64,
    expansion_add=256,
)
pegasus = Pegasus(config)
```

## Testing Strategy

Each module can be tested independently:

```python
# Test chunking
from pegasus.chunking import chunk_text
chunks = chunk_text("text", strategy="sentence")
assert len(chunks) > 0

# Test embeddings (requires OPENAI_API_KEY)
from pegasus.embeddings import EmbeddingProvider
emb = EmbeddingProvider("text-embedding-3-large")
vectors = emb.embed(["hello", "world"])
assert len(vectors) == 2

# Test storage
from pegasus.storage import MetadataStore
store = MetadataStore(":memory:")  # In-memory SQLite
chunk_id = store.insert_chunk({...})
assert chunk_id is not None
```

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Ingest per chunk | O(1) | Insert + single add to index |
| Search (vector) | O(log n) | HNSW with ef=expansion_search |
| Search (keyword) | O(log m) | FTS5 with B-tree |
| Search (hybrid) | O(log n + log m) | Both methods + RRF fusion |
| Save index | O(n) | Write all vectors to disk |

## Dependencies

### Core
- `usearch` — Vector index (HNSW)
- `openai` — Embedding API
- `sqlite3` — Built-in, no install

### Optional (for document loading)
- `langchain-community` — Document loaders
- `langchain-core` — Document model
- `beautifulsoup4` — HTML parsing
- `pymupdf` — PDF extraction
- `requests` — HTTP requests

### Development
- `tenacity` — Retry logic (for embeddings)
- `numpy` — Array operations (for USearch)

Install all:
```bash
pip install usearch "openai>=1.0.0" langchain-community langchain-core \
    beautifulsoup4 pymupdf requests tenacity numpy
```

## Future Improvements

- [ ] Async embedding batch processing
- [ ] Connection pooling for SQLite
- [ ] Multi-shard index support for billion-scale
- [ ] Custom distance metrics
- [ ] Incremental indexing (upsert)
- [ ] Built-in query expansion
- [ ] LLM-based re-ranking
