# Pegasus Product Sense & Value Proposition

## Why Pegasus Exists
Enterprise vector databases (e.g., Pinecone, Milvus, Qdrant) are powerful, but they bring massive operational overhead, infrastructure costs, and architectural complexity—especially for small-to-medium datasets (< 10M vectors) and edge/local apps. 

Developers are forced to deploy, secure, and manage independent databases for vector retrieval, metadata storing, and keyword retrieval, leading to fragmented queries and network overhead.

Pegasus offers a **unified local RAG storage engine** that delivers native-speed performance with zero external infrastructure.

## Value Pillars

### 1. Zero Infrastructure Overhead
No docker containers, cloud bills, cluster configs, or API keys for databases. It runs entirely in-process using SQLite and USearch, making deployment as simple as a python package import.

### 2. Best-of-Both-Worlds Search
Pure vector search is fantastic for conceptual queries but terrible for finding exact names, error codes, or IDs. Pure FTS (full-text search) is the opposite. By embedding both SQLite FTS5 and USearch, Pegasus provides out-of-the-box **hybrid search** using state-of-the-art Reciprocal Rank Fusion (RRF).

### 3. Native Speed & Low Footprint
Leveraging SIMD-accelerated HNSW indexing and memory-mapped files via USearch, Pegasus can serve sub-millisecond search requests while using 2x less memory with f16/bf16 quantized vectors.
