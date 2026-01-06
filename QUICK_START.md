# Quick Start Guide

## Installation

```bash
pip install usearch "openai>=1.0.0" langchain-community langchain-core \
    beautifulsoup4 pymupdf requests tenacity numpy
```

Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Basic Usage (5 minutes)

```python
from pegasus import create_pegasus, load_sources

# 1. Create engine
pegasus = create_pegasus("myrag.db", "myrag.usearch")

# 2. Load documents
docs = load_sources([
    "./documents/",      # Directory of PDFs/TXT/MD
    "https://example.com",  # Web page
    "./file.pdf",        # Single file
])

# 3. Ingest
stats = pegasus.ingest(docs, corpus="knowledge_base")
print(f"Indexed {stats['chunks']} chunks")

# 4. Search
results = pegasus.search("How do I...?", k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.content[:100]}...")

# 5. Close
pegasus.close()
```

## Search Modes

### Vector Search (Semantic)
Pure semantic similarity using HNSW:
```python
results = pegasus.search("machine learning", mode="vector", k=10)
```

### Keyword Search (Full-Text)
BM25-style keyword matching:
```python
results = pegasus.search("OAuth authentication", mode="keyword", k=10)
```

### Hybrid Search (Recommended)
Combines both with configurable weights:
```python
results = pegasus.search(
    "authentication setup guide",
    mode="hybrid",
    hybrid_alpha=0.7,  # 70% vector, 30% keyword
    k=10
)
```

## Configuration

### Default (Balanced)
```python
from pegasus import create_pegasus

pegasus = create_pegasus("db.db", "index.usearch")
```

### Memory-Optimized (f16 vectors)
```python
pegasus = create_pegasus("db.db", "index.usearch", dtype="f16")
# 2x smaller index, 99.5% recall
```

### High-Recall (larger HNSW)
```python
from pegasus import PegasusConfig, Pegasus

config = PegasusConfig(
    connectivity=64,       # More edges
    expansion_add=256,     # Better indexing
    expansion_search=128,  # Better search
    dtype="f32",          # Full precision
)
pegasus = Pegasus(config)
```

## Advanced Features

### Filtering by Corpus
```python
results = pegasus.search("query", corpus="knowledge_base")
```

### Custom Metadata Filtering
```python
results = pegasus.search(
    "query",
    filter_fn=lambda meta: meta.get("source") == "trusted"
)
```

### Managing Multiple Corpora
```python
# List all corpora
corpora = pegasus.list_corpora()
# [{'corpus': 'kb1', 'chunks': 1000, 'docs': 50}, ...]

# Delete a corpus
pegasus.delete_corpus("old_data")
```

### Get Engine Statistics
```python
stats = pegasus.get_stats()
# {
#     'index_size': 5000,
#     'embedding_model': 'text-embedding-3-large',
#     'dtype': 'f16',
#     'corpora': [...]
# }
```

## Component Usage

### Just Embeddings
```python
from pegasus import EmbeddingProvider

embedder = EmbeddingProvider("text-embedding-3-large")
vectors = embedder.embed(["text1", "text2", "text3"])
```

### Just Chunking
```python
from pegasus import chunk_text

chunks = chunk_text(
    "long text",
    max_chars=2000,
    overlap_chars=200,
    strategy="sentence"
)
```

### Just Document Loading
```python
from pegasus import load_sources

docs = load_sources([
    "./pdfs/",
    "https://example.com",
    "./file.txt"
])
```

### Just Vector Index
```python
from pegasus import VectorIndexManager

index = VectorIndexManager(
    "index.usearch",
    embedding_dim=3072,
    dtype="f16"
)
index.add(1, embedding_vector)
matches = index.search(query_vector, k=10)
```

### Just Metadata Storage
```python
from pegasus import MetadataStore

store = MetadataStore("pegasus.db")

# Insert chunk
chunk_id = store.insert_chunk({
    "corpus": "kb",
    "doc_id": "doc1",
    "chunk_index": 0,
    "content": "text...",
    "content_hash": "...",
    "source": "pdf",
    "title": "Title",
    "page": 1,
    "metadata_json": '{"key": "value"}'
})

# Search FTS
results = store.search_fts("search query", k=10)

# Get chunk
chunk = store.get_chunk(chunk_id)
```

## Common Patterns

### Multi-Corpus RAG
```python
pegasus = create_pegasus("db.db", "index.usearch")

# Load different sources
docs_kb = load_sources(["./knowledge/"])
docs_faq = load_sources(["./faq/"])

pegasus.ingest(docs_kb, corpus="knowledge")
pegasus.ingest(docs_faq, corpus="faq")

# Search one corpus
kb_results = pegasus.search("query", corpus="knowledge", k=5)
```

### Quality-Aware Search
```python
# Stricter filtering
results = pegasus.search(
    "query",
    mode="hybrid",
    hybrid_alpha=0.8,  # Favor semantic
    k=20,
    filter_fn=lambda m: m.get("source") in ["official", "verified"]
)
results = results[:5]  # Top 5 best
```

### Batch Ingest
```python
for batch in batches:
    docs = load_sources(batch)
    pegasus.ingest(docs, corpus=f"batch_{i}")
    pegasus.save()  # Checkpoint
```

### Export Search Results
```python
import json

results = pegasus.search("query", k=10)
json_results = [
    {
        "chunk_id": r.chunk_id,
        "content": r.content,
        "score": r.score,
        "metadata": r.metadata,
    }
    for r in results
]
print(json.dumps(json_results, indent=2))
```

## Troubleshooting

### API Key Missing
```python
# Solution 1: Set env var
export OPENAI_API_KEY="sk-..."

# Solution 2: Pass directly
pegasus = create_pegasus("db.db", "index.usearch", 
                         openai_api_key="sk-...")
```

### Memory Usage High
```python
# Use f16 instead of f32
pegasus = create_pegasus("db.db", "index.usearch", dtype="f16")

# Or reduce HNSW parameters
config = PegasusConfig(connectivity=16, expansion_add=64)
```

### Search Too Slow
```python
# Increase search parameters
config = PegasusConfig(expansion_search=256)

# Or use keyword search for exact matches
results = pegasus.search("exact phrase", mode="keyword")
```

### Duplicates in Results
```python
# Chunks are deduped by content_hash
# If seeing duplicates, check if intentional overlap
# Reduce chunk_overlap in config

config = PegasusConfig(chunk_overlap=32)  # Lower overlap
```

## Performance Tips

1. **Use f16 dtype** — 2x smaller index, negligible recall loss
2. **Batch embed** — `embed()` takes list of texts
3. **Filter early** — Use corpus/filter_fn to reduce search space
4. **Tune hybrid_alpha** — 0.7 is good default, adjust for your data
5. **Right chunk size** — 512 tokens ≈ 2000 chars, adjust for your domain

## Next Steps

- Read [STRUCTURE.md](STRUCTURE.md) for deep dive
- Check [README.md](README.md) for benchmarks and architecture
- See [AGENTS.md](AGENTS.md) for development workflow
- Run demo: `python -m pegasus.cli`
