# Pegasus Frontend & REST API Integration

Pegasus packages a FastAPI REST server to serve as an HTTP backend for web or native frontends.

## 1. REST API Endpoints

The server runs on `uvicorn pegasus.api:main` and exposes the following endpoints:

### `POST /ingest`
Ingests a list of documents or raw text inputs into a corpus.

- **Request Body**:
  ```json
  {
    "documents": [
      {
        "text": "Pegasus is highly modular.",
        "metadata": {"category": "tech"},
        "doc_id": "optional_id"
      }
    ],
    "corpus": "knowledge_base"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "chunks": 1,
    "skipped": 0,
    "docs": 1
  }
  ```

### `POST /search`
Executes vector, keyword, or hybrid searches.

- **Request Body**:
  ```json
  {
    "query": "modularity",
    "corpus": "knowledge_base",
    "mode": "hybrid",
    "k": 5,
    "hybrid_alpha": 0.7
  }
  ```
- **Response**:
  ```json
  {
    "query": "modularity",
    "results": [
      {
        "chunk_id": 1,
        "doc_id": "optional_id",
        "content": "Pegasus is highly modular.",
        "score": 0.84,
        "metadata": {"category": "tech"}
      }
    ]
  }
  ```

### `GET /stats`
Retrieves storage and index metrics.

- **Response**:
  ```json
  {
    "vector_count": 1,
    "corpora_count": 1,
    "sqlite_database": "pegasus.db"
  }
  ```
