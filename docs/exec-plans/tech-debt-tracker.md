# Pegasus Technical Debt Tracker

This document tracks identified areas of technical debt, architectural bottlenecks, and engineering tasks that require future attention.

## Active & Pending Tech Debt

### 1. Synchronous Embeddings Pipeline
- **Description**: Generating embeddings is currently fully synchronous, which blocks thread workers during bulk ingestion.
- **Impact**: Ingestion speed is bottlenecked by network round-trip latencies.
- **Planned Refactor**: Add async embedding generation support with connection semaphores (tracked in `pegasus-zg5`).

### 2. Missing Prometheus Metrics
- **Description**: Observing latency, search modes, cache hits, and SQLite health is currently done through standard logs.
- **Impact**: Hard to monitor in real-world production environments.
- **Planned Refactor**: Integrate standard observability metrics (tracked in `pegasus-kr2`).

### 3. Tree-sitter Code Chunking
- **Description**: Our chunker uses sentence/paragraph splitters, which are suboptimal for splitting programming language files (Python, JavaScript, etc.).
- **Impact**: Code chunks lose semantic structure.
- **Planned Refactor**: Introduce AST-aware semantic chunking (tracked in `pegasus-4w6`).

## Resolved Tech Debt

- **Monolith to Modular Package**: Refactored `pegasus_v2.py` monolith into 10 highly focused modules under `pegasus/` with clean interface separation.
- **Redundant Import Overhead**: Optimized lazy imports in `pegasus/integration.py` to check for `sentence_transformers` with `importlib.util` instead of throwing `ImportError`.
