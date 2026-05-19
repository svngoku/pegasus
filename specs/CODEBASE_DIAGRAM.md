# Pegasus Codebase Architecture Diagram

**Project:** Pegasus v2 — High-Performance RAG Engine  
**Generated:** 2026-02-14  
**Purpose:** Executive-friendly overview of codebase organization and architecture

---

## 1. System Context Diagram

```mermaid
flowchart LR
    subgraph External["External Systems"]
        user["End User / Developer"]
        openai["OpenAI API"]
        hf["HuggingFace Hub"]
        jina["Jina AI API"]
    end

    subgraph Pegasus["Pegasus RAG Engine"]
        cli["CLI / Script"]
        api["REST API\n(FastAPI)"]
        core["Core Engine\n(pegasus.py)"]
    end

    subgraph Storage["Data Layer"]
        usearch["USearch\nVector Index"]
        sqlite["SQLite\n+ FTS5"]
        cache["LRU Cache\n(In-memory)"]
    end

    user --> cli
    user --> api
    api --> core
    core --> openai
    core --> hf
    core --> jina
    core --> usearch
    core --> sqlite
    core --> cache
```

### Code Anchors

| Node | Path/Module | Description |
|------|-------------|-------------|
| `cli` | `pegasus/cli.py` | Command-line interface entry point |
| `api` | `pegasus/api.py` | FastAPI REST API server |
| `core` | `pegasus/pegasus.py` | Main orchestrator class |
| `openai` | `pegasus/embeddings.py` | OpenAI embedding provider |
| `usearch` | `pegasus/index.py` | USearch HNSW vector index |
| `sqlite` | `pegasus/storage.py` | SQLite + FTS5 metadata storage |

---

## 2. Codebase Architecture Diagram

```mermaid
flowchart TB
    subgraph Ingestion["Ingestion Pipeline"]
        loaders["loaders.py\nDocument Loaders"]
        chunking["chunking.py\nText Chunking"]
        embeddings["embeddings.py\nEmbedding Provider"]
    end

    subgraph Core["Core Domain"]
        pegasus["pegasus.py\nPegasus Orchestrator"]
        config["config.py\nConfiguration"]
        models["models.py\nData Models"]
    end

    subgraph Search["Search Pipeline"]
        search["search.py\nSearch Dispatcher"]
        reranker["reranker.py\nReranker"]
    end

    subgraph Storage["Data Layer"]
        index["index.py\nVector Index\n(USearch HNSW)"]
        storage["storage.py\nSQLite + FTS5"]
    end

    loaders --> chunking
    chunking --> embeddings
    embeddings --> pegasus
    pegasus --> index
    pegasus --> storage
    pegasus --> search
    search --> reranker
```

### Code Anchors

| Node | Path | Responsibility |
|------|------|----------------|
| `loaders` | `pegasus/loaders.py` | Multi-source document loading (PDF, MD, URL, TXT) |
| `chunking` | `pegasus/chunking.py` | Sentence-aware text splitting with configurable overlap |
| `embeddings` | `pegasus/embeddings.py` | OpenAI/HF/Jina embedding providers with retry + batch |
| `pegasus` | `pegasus/pegasus.py` | Main orchestrator - threading, factory methods |
| `config` | `pegasus/config.py` | Immutable configuration schema |
| `models` | `pegasus/models.py` | Data classes (PegasusDoc, SearchResult) |
| `search` | `pegasus/search.py` | Search dispatcher (vector, keyword, hybrid modes) |
| `reranker` | `pegasus/reranker.py` | Result reranking |
| `index` | `pegasus/index.py` | USearch HNSW vector index manager |
| `storage` | `pegasus/storage.py` | SQLite + FTS5 metadata storage |

---

## 3. Request Lifecycle - Ingestion

```mermaid
sequenceDiagram
    participant User
    participant API as api.py
    participant Core as pegasus.py
    participant Loader as loaders.py
    participant Chunk as chunking.py
    participant Embed as embeddings.py
    participant Index as index.py
    participant Store as storage.py

    User->>API: ingest(texts, metadata)
    API->>Core: Pegasus.ingest()
    Core->>Loader: load_sources()
    Loader-->>Core: PegasusDoc[]
    Core->>Chunk: chunk_text()
    Chunk-->>Core: Chunks[]
    Core->>Embed: get_embeddings()
    Embed->>Embed: OpenAI API
    Embed-->>Core: vectors[]
    Core->>Index: add_vectors()
    Index-->>Core: chunk_ids[]
    Core->>Store: store_metadata()
    Store-->>Core: committed
    Core-->>API: IngestResponse
    API-->>User: {chunks, skipped, docs}
```

---

## 4. Request Lifecycle - Search

```mermaid
sequenceDiagram
    participant User
    participant API as api.py
    participant Core as pegasus.py
    participant Embed as embeddings.py
    participant Search as search.py
    participant Index as index.py
    participant Store as storage.py

    User->>API: search(query, mode, k)
    API->>Core: Pegasus.search()
    Core->>Embed: get_embeddings(query)
    Embed-->>Core: query_vector
    Core->>Search: dispatch_search()
    
    alt Vector Search
        Core->>Index: vector_search()
        Index-->>Core: vector_results[]
    end
    
    alt Keyword Search
        Core->>Store: fts5_search()
        Store-->>Core: keyword_results[]
    end
    
    alt Hybrid Search
        Search-->>Search: RRF combination
    end
    
    Search-->>Core: ranked_results[]
    Core-->>API: SearchResponse
    API-->>User: {results[], query, mode, count}
```

---

## 5. Data Storage Architecture

```mermaid
flowchart LR
    subgraph Vector["Vector Storage"]
        usearch["USearch Index\n- HNSW graph\n- f16/bf16 vectors\n- Memory-mapped"]
    end

    subgraph Metadata["Metadata Storage"]
        sqlite["SQLite Database\n- chunk_metadata table\n- FTS5 full-text index\n- Corpus filtering"]
    end

    subgraph Cache["In-Memory Cache"]
        lru["LRU Cache\n- Embedding cache\n- Thread-safe (RLock)"]
    end

    usearch <-->|"chunk_id"| sqlite
    lru <-->|"query embedding"| usearch
```

---

## 6. Optional: Deployment View

```mermaid
flowchart LR
    subgraph Client["Clients"]
        web["Web App"]
        script["Python Script"]
        cli["CLI Tool"]
    end

    subgraph Server["Pegasus Service"]
        fastapi["FastAPI\n(Uvicorn)"]
        engine["Pegasus Engine"]
    end

    subgraph External["External Services"]
        openai_api["OpenAI API\n(Embeddings)"]
    end

    web --> fastapi
    script --> engine
    cli --> engine
    fastapi --> engine
    engine --> openai_api
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| `subgraph` | Logical grouping of related components |
| `-->` | Data/control flow direction |
| `A <--> B` | Bidirectional relationship |
| Solid border | Internal Pegasus components |
| Dashed border | External systems/services |

---

## Key Technologies

| Component | Technology |
|-----------|------------|
| Vector Search | USearch (HNSW, SIMD-accelerated) |
| Metadata Storage | SQLite + FTS5 |
| Embedding Providers | OpenAI, HuggingFace, Jina AI |
| REST API | FastAPI + Uvicorn |
| Thread Safety | Python RLock |
| Retry Logic | Tenacity (exponential backoff) |

---

## Unknowns / Assumptions

- **No message queues detected** - This is a synchronous RAG engine, no async job processing
- **Single-node deployment** - No distributed architecture in current version
- **No authentication layer** - API assumes trusted environment
- **Local file storage only** - No cloud storage integration (S3, etc.)

To verify: Check `pegasus/integration.py` for any additional external integrations.
