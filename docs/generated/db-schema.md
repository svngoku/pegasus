# Pegasus Database & Storage Schema

Pegasus uses a hybrid storage model combining a **SQLite Database** for full-text search and metadata, alongside a **USearch HNSW Index** for vector similarity.

## 1. SQLite Database Schema

The database contains a main table for chunks and an FTS5 virtual table for keyword retrieval.

### `chunks` Table
Holds raw chunk content, document structure, hashes for deduplication, and JSON metadata.

| Column | Type | Constraints | Description |
|---|---|---|---|
| `id` | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique chunk identifier (mapped to HNSW key) |
| `corpus` | TEXT | NOT NULL | Collection/corpus identifier |
| `doc_id` | TEXT | NOT NULL | ID of parent document (stable sha256 hash) |
| `content` | TEXT | NOT NULL | Raw text content of the chunk |
| `content_hash` | TEXT | UNIQUE NOT NULL | SHA256 hash of `content` (enforces deduplication) |
| `metadata_json` | TEXT | NULL | JSON string containing arbitrary document metadata |
| `created_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Row creation time |
| `updated_at` | TIMESTAMP | DEFAULT CURRENT_TIMESTAMP | Row last modified time |

### `chunks_fts` Virtual Table (FTS5)
A full-text search virtual table powered by SQLite's FTS5 extension.

- **Columns**: `content`
- **Configuration**: Uses content-less FTS5 indexes mapped directly to the `chunks.id` rowid to save disk space.
- **Search Method**: BM25 ranking via `MATCH` queries.

---

## 2. USearch Index File (`.usearch`)

USearch persists its HNSW (Hierarchical Navigable Small World) index to a single memory-mapped binary file.

- **Key (ID)**: Integer matching `chunks.id`.
- **Dimensions**: Configurable (defaults to 3072 for `text-embedding-3-large`).
- **Precision**: Supported types include `f32` (standard), `f16` (recommended, half-memory), `bf16`, and `i8`.
- **Metric**: Cosine similarity (`cos`), Inner Product (`ip`), or Euclidean distance (`l2sq`).
