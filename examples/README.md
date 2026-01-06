# Pegasus Examples

This directory contains example scripts demonstrating various Pegasus features.

## Prerequisites

```bash
# Install Pegasus
uv add pegasus-rag
# or
pip install pegasus-rag

# Set API keys (as needed)
export OPENAI_API_KEY=sk-...      # Required for OpenAI
export HF_TOKEN=hf_...            # Optional for HuggingFace
export JINA_API_KEY=jina_...      # Required for Jina AI
```

## Examples

### 01_basic_rag.py - Basic RAG Pipeline

Demonstrates the core RAG workflow:
- Creating a Pegasus instance
- Ingesting documents with progress callbacks
- Searching with vector, keyword, and hybrid modes
- Exporting/importing corpora for backup

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/01_basic_rag.py
```

### 02_multi_provider.py - Multi-Provider Embeddings

Demonstrates using different embedding providers:
- HuggingFace sentence-transformers (local, free)
- OpenAI embeddings (API)
- Jina AI embeddings (API)
- LLM-based re-ranking for improved relevance

```bash
# HuggingFace only (no API key needed)
uv add sentence-transformers
uv run python examples/02_multi_provider.py

# With all providers
export OPENAI_API_KEY=sk-...
export JINA_API_KEY=jina_...
uv run python examples/02_multi_provider.py
```

## Quick Usage Examples

### From PyPI (after `pip install pegasus-rag`)

```python
from pegasus import create_client, quick_search

# Quick one-shot search
results = quick_search(
    "machine learning",
    ["AI is the future", "ML enables predictions", "Python is popular"],
)
print(results[0]["content"])

# Full client usage
with create_client(provider="huggingface") as client:
    client.ingest(["Document 1", "Document 2"])
    results = client.search("query")
```

### With Context Manager

```python
from pegasus.integration import PegasusClient, EmbeddingConfig

config = EmbeddingConfig(provider="huggingface", model="all-MiniLM-L6-v2")

with PegasusClient(embedding=config) as client:
    client.ingest(["Hello world", "Goodbye world"])
    results = client.ask("greeting")
    print(results)
```

## Available Providers

| Provider | API Key | Local | Cost |
|----------|---------|-------|------|
| OpenAI | `OPENAI_API_KEY` | No | ~$0.13/1M tokens |
| HuggingFace | Optional `HF_TOKEN` | Yes | Free |
| Jina AI | `JINA_API_KEY` | No | Free tier available |

## More Information

- [README.md](../README.md) - Main documentation
- [QUICK_START.md](../QUICK_START.md) - Quick start guide
- [ARCHITECTURE.md](../ARCHITECTURE.md) - Technical architecture
