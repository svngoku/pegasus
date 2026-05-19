# Pegasus Project Plans & Roadmap

This document outlines upcoming, in-progress, and completed execution plans for the Pegasus RAG engine.

## 🏁 Completed Milestones
- **Modular Packaging Refactor**: Successfully split the monolithic `pegasus_v2.py` into cohesive Python modules inside `pegasus/`.
- **HuggingFace & Jina AI Support**: Added offline execution mode and Jina AI integration.
- **REST Wrapper**: Built high-performance FastAPI server.

## 🚀 Active Roadmap

### 1. Ingestion Speed Up via Async API (Medium Priority)
- **Problem**: Generating embeddings for large document collections is limited by synchronous single-threaded network calls.
- **Solution**: Implement concurrent embedding retrieval utilizing Python's `asyncio` and semaphores to maximize ingest throughput.

### 2. Semantic AST Chunking (Medium Priority)
- **Problem**: Standard sentence and paragraph splitting breaks context boundaries in structured formats (code files, YAML, JSON, multi-level Markdown).
- **Solution**: Incorporate tree-sitter or AST-aware chunkers to respect logical boundaries inside programming and markup files.

### 3. Basic observability & Latency tracking (Low Priority)
- **Problem**: Monitoring engine performance currently relies on stdout/stderr logging.
- **Solution**: Package structured observability using `structlog` alongside standard metrics for latency percentiles and cache efficiency.
