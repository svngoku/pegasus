# Pegasus Project Map

## Quick Navigation

### 📚 For Users
Start here for using Pegasus:
1. [README.md](README.md) — Overview, features, benchmarks
2. [QUICK_START.md](QUICK_START.md) — 5-minute guide & examples
3. [README.md § Configuration](README.md#configuration) — Tuning HNSW
4. [README.md § Search Modes](README.md#search-modes) — Vector, keyword, hybrid

### 🏗️ For Developers
Understanding the codebase:
1. [STRUCTURE.md](STRUCTURE.md) — Module reference & responsibilities
2. [ARCHITECTURE.md](ARCHITECTURE.md) — Design, data flow, extension points
3. Source code in `pegasus/` — Read modules top-to-bottom
4. [AGENTS.md](AGENTS.md) — Development workflow & commands

### 🔧 For Contributors
Making changes:
1. Read [AGENTS.md](AGENTS.md) for development practices
2. Follow [ARCHITECTURE.md § Separation of Concerns](ARCHITECTURE.md#separation-of-concerns)
3. Each module is independent — test in isolation
4. Update related docs when adding features

---

## File Structure

```
pegasus/                          ← Package root
├── README.md                     # Main documentation
├── QUICK_START.md               # 5-minute examples
├── STRUCTURE.md                 # Module reference
├── ARCHITECTURE.md              # Design & internals
├── PROJECT_MAP.md              # This file
├── AGENTS.md                    # Development guide
├── setup.py                     # Installation config
├── pegasus_v2.py               # Original monolithic (legacy)
│
└── pegasus/                      ← Main package
    ├── __init__.py              # Public API exports
    ├── config.py                # Config dataclass
    ├── models.py                # Data classes
    ├── loaders.py               # Multi-source loading
    ├── chunking.py              # Text splitting
    ├── embeddings.py            # OpenAI API
    ├── index.py                 # Vector index
    ├── storage.py               # SQLite + FTS5
    ├── search.py                # Search engine
    ├── pegasus.py               # Main orchestrator
    └── cli.py                   # Demo
```

---

## Key Concepts

### Components (Modular)
```
Pegasus (Main)
├── EmbeddingProvider ← OpenAI API
├── VectorIndexManager ← USearch HNSW
├── MetadataStore ← SQLite + FTS5
└── SearchEngine ← Dispatcher (3 modes)
```

### Search Modes
| Mode | Best For | Speed | Recall |
|------|----------|-------|--------|
| **Vector** | Semantic similarity | Fast | Medium |
| **Keyword** | Exact matches | Fast | Low |
| **Hybrid** | Balanced | Slower | High |

### Data Models
```
PegasusDoc          SearchResult
├── text            ├── chunk_id
├── metadata        ├── doc_id
└── doc_id          ├── content
                    ├── score
                    └── metadata
```

---

## Common Tasks

### "I want to use Pegasus in my app"
→ [QUICK_START.md](QUICK_START.md)

### "I want to understand the codebase"
→ [STRUCTURE.md](STRUCTURE.md) then [ARCHITECTURE.md](ARCHITECTURE.md)

### "I want to add a new feature"
→ Read [ARCHITECTURE.md § Extension Points](ARCHITECTURE.md#extension-points)

### "I want to optimize performance"
→ [ARCHITECTURE.md § Performance Tuning](ARCHITECTURE.md#performance-tuning)

### "I want to customize search"
→ [ARCHITECTURE.md § Component Diagram](ARCHITECTURE.md#component-diagram)

### "I want to deploy to production"
→ [ARCHITECTURE.md § Deployment Patterns](ARCHITECTURE.md#deployment-patterns)

### "I want to contribute"
→ [AGENTS.md](AGENTS.md)

---

## Module Purpose Matrix

| Module | Input | Output | When to Change |
|--------|-------|--------|---|
| `config.py` | Settings | Settings object | New parameter needed |
| `models.py` | Fields | Typed objects | Add result fields |
| `loaders.py` | URLs/paths | PegasusDoc[] | Support new format |
| `chunking.py` | Text | Chunks[] | Improve splitting |
| `embeddings.py` | Text | Vectors[] | Change embedding API |
| `index.py` | Vectors | Index operations | Switch vector DB |
| `storage.py` | Metadata | DB operations | Change schema |
| `search.py` | Query | Results[] | Add search mode |
| `pegasus.py` | Everything | High-level API | Change flow |

---

## How to Read the Code

### Linear Reading (Recommended)
1. Start with [config.py](pegasus/config.py) — All settings in one place
2. Then [models.py](pegasus/models.py) — Data structures
3. Then [chunking.py](pegasus/chunking.py) — Simple utility
4. Then [loaders.py](pegasus/loaders.py) — Input handling
5. Then [embeddings.py](pegasus/embeddings.py) — API integration
6. Then [index.py](pegasus/index.py) — Vector storage
7. Then [storage.py](pegasus/storage.py) — Metadata storage
8. Then [search.py](pegasus/search.py) — Query logic
9. Finally [pegasus.py](pegasus/pegasus.py) — Orchestration

### By Concern
- **Data**: config.py, models.py
- **Input**: loaders.py, chunking.py
- **APIs**: embeddings.py
- **Storage**: index.py, storage.py
- **Query**: search.py
- **Glue**: pegasus.py

### By Dependency
```
config.py ←─┐
             │
models.py ←─┼─────┐
             │     │
loaders.py ←┴─┐   │
              │   │
chunking.py   │   │
(independent) │   │
              │   │
embeddings.py ├─→ pegasus.py
              │   ↑
index.py  ────┤   │
              │   │
storage.py ───┤   │
              │   │
search.py  ←──┴──→┤
                  └─→ cli.py
```

---

## Testing Approach

### Unit Tests (per module)
```python
# test_chunking.py
from pegasus.chunking import chunk_text

def test_chunk_text_sentence():
    chunks = chunk_text("a. b. c.", strategy="sentence")
    assert len(chunks) == 3

# test_config.py
from pegasus.config import PegasusConfig

def test_config_defaults():
    config = PegasusConfig()
    assert config.dtype == "f16"
```

### Integration Tests
```python
# test_integration.py (requires OPENAI_API_KEY)
from pegasus import create_pegasus, PegasusDoc

def test_ingest_and_search():
    pegasus = create_pegasus(":memory:", "/tmp/test.usearch")
    doc = PegasusDoc(text="hello world")
    stats = pegasus.ingest([doc])
    assert stats["chunks"] > 0
    
    results = pegasus.search("hello", k=5)
    assert len(results) > 0
```

### Component Tests
```python
# test_storage.py (no external deps)
from pegasus.storage import MetadataStore

def test_insert_and_get():
    store = MetadataStore(":memory:")  # In-memory DB
    chunk_id = store.insert_chunk({...})
    chunk = store.get_chunk(chunk_id)
    assert chunk is not None
```

---

## Configuration Presets

### Development
```python
from pegasus import PegasusConfig, Pegasus

config = PegasusConfig(
    dtype="f32",              # Full precision
    expansion_add=64,         # Faster indexing
    expansion_search=32,      # Faster search
    chunk_size=512,           # Standard
)
pegasus = Pegasus(config)
```

### Production (Memory-Optimized)
```python
config = PegasusConfig(
    dtype="f16",              # 2x smaller
    connectivity=32,          # Balanced
    expansion_add=128,        # Good quality
    expansion_search=64,      # Balanced search
)
pegasus = Pegasus(config)
```

### Production (High-Recall)
```python
config = PegasusConfig(
    dtype="f32",              # Full precision
    connectivity=64,          # More edges
    expansion_add=256,        # Better indexing
    expansion_search=128,     # Better search
)
pegasus = Pegasus(config)
```

### Production (High-Speed)
```python
config = PegasusConfig(
    dtype="i8",               # Smallest (quantized)
    connectivity=16,          # Fewer edges
    expansion_add=64,         # Fast indexing
    expansion_search=32,      # Fast search
)
pegasus = Pegasus(config)
```

---

## API Quick Reference

### Ingestion
```python
# Load documents
docs = load_sources(["./docs", "https://example.com"])

# Ingest
stats = pegasus.ingest(docs, corpus="kb", show_progress=True)
# → {"chunks": 1000, "skipped": 50, "docs": 20}
```

### Search
```python
# Vector search (semantic)
results = pegasus.search("query", k=10, mode="vector")

# Keyword search (full-text)
results = pegasus.search("query", k=10, mode="keyword")

# Hybrid search (RRF)
results = pegasus.search("query", k=10, mode="hybrid", hybrid_alpha=0.7)

# With corpus filter
results = pegasus.search("query", corpus="kb", k=10)

# With metadata filter
results = pegasus.search(
    "query",
    filter_fn=lambda m: m.get("source") == "trusted"
)
```

### Management
```python
# List corpora
corpora = pegasus.list_corpora()

# Delete corpus
pegasus.delete_corpus("old_kb")

# Get stats
stats = pegasus.get_stats()

# Save
pegasus.save()

# Close
pegasus.close()
```

---

## Troubleshooting

### Import Error
```python
# Fix: Install dependencies
pip install pegasus-rag

# Or from source
pip install -e .
```

### API Key Missing
```python
# Fix: Set env var
export OPENAI_API_KEY="sk-..."

# Or pass directly
pegasus = create_pegasus(..., openai_api_key="sk-...")
```

### Memory Usage High
```python
# Use f16 instead of f32
config = PegasusConfig(dtype="f16")
```

### Search Too Slow
```python
# Reduce search parameter
config = PegasusConfig(expansion_search=32)
```

---

## Performance Targets

| Operation | Target | Notes |
|-----------|--------|-------|
| Ingest 1000 chunks | < 30s | Includes embedding API calls |
| Search 10k vectors | < 10ms | Vector search |
| Search 10k vectors | < 50ms | Hybrid search |
| Index 1M vectors | 5-10 GB | f16 dtype |

---

## Next Steps

1. **New users**: Start with [QUICK_START.md](QUICK_START.md)
2. **Developers**: Read [STRUCTURE.md](STRUCTURE.md) then dive into code
3. **Contributors**: Follow [AGENTS.md](AGENTS.md)
4. **Questions**: Check [ARCHITECTURE.md](ARCHITECTURE.md) § Extension Points

---

## Document Map

| Document | Audience | Purpose |
|----------|----------|---------|
| README.md | Everyone | Overview, features, benchmarks |
| QUICK_START.md | Users | Copy-paste examples |
| STRUCTURE.md | Developers | Module reference |
| ARCHITECTURE.md | Developers | Design & internals |
| PROJECT_MAP.md | Everyone | Navigation (this file) |
| AGENTS.md | Contributors | Development workflow |

---

**Last Updated:** 2025-01-06
**Version:** 2.0.0
