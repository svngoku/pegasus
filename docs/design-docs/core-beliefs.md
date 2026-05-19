# Pegasus Core Beliefs & Design Philosophy

## 1. Composition Over Inheritance
We believe in building small, highly focused components and combining them through composition rather than complex class hierarchies. This ensures that:
- Every component is easily testable in isolation.
- Parts of the pipeline can be swapped out (e.g., custom embedder, alternative storage backend) with zero modifications to the core orchestrator.

## 2. Monolithic and Modular
While a monolithic single file is easy to read initially, it quickly becomes unmaintainable. We maintain absolute modularity in the package structure (`pegasus/` directory containing focused modules like `chunking`, `embeddings`, `index`, `storage`, `search`), while exposing a simple, unified public API (`create_pegasus`) for users who want a "just works" experience.

## 3. Simplicity & Native Speed
Avoid unnecessary abstractions and complex distributed query engines. By coupling USearch (a high-performance, single-header native HNSW vector index) with SQLite (a highly optimized local SQL database), we achieve performance comparable to specialized vector databases with zero operational overhead, zero external dependencies, and minimal memory foot-print.

## 4. Half-Precision Correctness
Modern vector databases waste massive memory storing full-precision floats (f32). We native-ly support f16 and bf16 vector types, cutting memory consumption in half while retaining identical recall rates.

## 5. Thread Safety First
RAG pipelines frequently serve concurrent search requests and background ingestion. All mutable storage and indexing operations in Pegasus are protected by an `RLock` (reentrant lock) to guarantee thread safety and prevent database corruption.
