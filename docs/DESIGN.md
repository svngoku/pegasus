# Pegasus Design Specification

Pegasus combines a lightning-fast vector search index with a resilient metadata database, orchestrating them through a clean modular Python API.

```
                  ┌──────────────────────┐
                  │       Pegasus        │
                  │    (Orchestrator)    │
                  └──────────┬───────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│EmbeddingProvider│ │MetadataStore    │ │VectorIndexMan.  │
│ (APIs/Caching)  │ │ (SQLite/FTS5)   │ │ (USearch HNSW)  │
└─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Architectural Design Pattern

### 1. Unified Client Interface
The main class `Pegasus` acts as an orchestrator. It delegates embedding generation, database inserts, index modifications, and search dispatches to specialized components.

### 2. Double DB Sync Ingestion
When ingesting documents:
1. Texts are split into chunks using `chunk_text()`.
2. Cached embeddings are fetched or generated via `EmbeddingProvider`.
3. Metadata is written to `MetadataStore` in SQLite. SQLite returns a unique, autoincremented integer `id`.
4. The generated embedding is added to `VectorIndexManager` using this `id` as the key.

This ensures a perfect 1:1 key-to-metadata association with minimal overhead.

### 3. Dispatched Search Mode
Search queries can be run in `vector`, `keyword`, or `hybrid` mode:
- **Vector**: Search USearch using query embedding and look up matching texts in SQLite.
- **Keyword**: Query the SQLite FTS5 index directly with exact keywords.
- **Hybrid**: Query both vector index and keyword database, combining results using Reciprocal Rank Fusion (RRF) with a tunable `alpha` weight.
