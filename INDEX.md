# Pegasus Project Index

## 📚 Documentation Files

| File | Purpose | Audience | Read Time |
|------|---------|----------|-----------|
| [README.md](README.md) | Main documentation, features, benchmarks | Everyone | 10 min |
| [QUICK_START.md](QUICK_START.md) | 5-minute guide with copy-paste examples | Users | 5 min |
| [STRUCTURE.md](STRUCTURE.md) | Detailed module reference and API | Developers | 15 min |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Design, data flow, extension points | Developers | 20 min |
| [PROJECT_MAP.md](PROJECT_MAP.md) | Navigation guide and quick reference | Everyone | 10 min |
| [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) | What changed in the reorg | Contributors | 10 min |
| [AGENTS.md](AGENTS.md) | Development workflow and commands | Contributors | 10 min |
| [INDEX.md](INDEX.md) | This file - quick navigation | Everyone | 2 min |

## 🐍 Code Structure

```
pegasus/                          # Main package
├── __init__.py                  # Public API exports
├── config.py                    # Configuration (PegasusConfig)
├── models.py                    # Data types (PegasusDoc, SearchResult)
├── loaders.py                   # Document loading (load_sources)
├── chunking.py                  # Text splitting (chunk_text)
├── embeddings.py                # OpenAI API wrapper
├── index.py                     # USearch HNSW wrapper
├── storage.py                   # SQLite + FTS5 wrapper
├── search.py                    # Search engine (3 modes)
├── pegasus.py                   # Main orchestrator (Pegasus class)
└── cli.py                       # Demo CLI entry point
```

## 🎯 Quick Navigation

### "I want to..."

| Goal | Start Here |
|------|-----------|
| **Use Pegasus in my app** | [QUICK_START.md](QUICK_START.md) |
| **Understand architecture** | [ARCHITECTURE.md](ARCHITECTURE.md) |
| **Look up a module** | [STRUCTURE.md](STRUCTURE.md) |
| **Extend/customize it** | [ARCHITECTURE.md](#extension-points) |
| **Set up development** | [AGENTS.md](AGENTS.md) |
| **Find a specific file** | [PROJECT_MAP.md](#file-structure) |
| **See all changes** | [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) |

## 📦 What's Included

### Core Components
- ✅ **Vector Search** — USearch HNSW with f16/bf16 support
- ✅ **Full-Text Search** — SQLite FTS5 with BM25 ranking
- ✅ **Hybrid Search** — Reciprocal Rank Fusion (RRF) fusion
- ✅ **Document Loading** — URLs, PDFs, TXT, MD, directories
- ✅ **Smart Chunking** — Sentence-aware, paragraph-aware, fixed
- ✅ **Embeddings** — OpenAI API with retry and batching
- ✅ **Metadata Storage** — SQLite with deduplication
- ✅ **Thread Safety** — RLock for concurrent operations

### Tools & Integrations
- ✅ **OpenAI API** — text-embedding-3-large and others
- ✅ **USearch** — Native HNSW with SIMD acceleration
- ✅ **SQLite** — Built-in, no external DB required
- ✅ **LangChain** — Document loaders for multiple formats

## 🚀 Getting Started (2 minutes)

```bash
# 1. Install
pip install usearch "openai>=1.0.0" langchain-community langchain-core \
    beautifulsoup4 pymupdf requests tenacity numpy

# 2. Set API key
export OPENAI_API_KEY="sk-..."

# 3. Write code
python << 'EOF'
from pegasus import create_pegasus, load_sources

pegasus = create_pegasus("db.db", "index.usearch")
docs = load_sources(["./documents/"])
pegasus.ingest(docs, corpus="knowledge")
results = pegasus.search("How do I...?", mode="hybrid", k=5)
for r in results:
    print(f"[{r.score:.2f}] {r.content[:100]}...")
pegasus.close()
EOF
```

## 📖 Reading Paths

### Path 1: User (5 minutes)
1. [README.md](README.md) — Overview
2. [QUICK_START.md](QUICK_START.md) — Examples
3. Start building!

### Path 2: Developer (30 minutes)
1. [QUICK_START.md](QUICK_START.md) — Get familiar
2. [STRUCTURE.md](STRUCTURE.md) — Learn modules
3. [ARCHITECTURE.md](ARCHITECTURE.md) — Understand design
4. Read source code in pegasus/ folder

### Path 3: Contributor (1 hour)
1. [AGENTS.md](AGENTS.md) — Development setup
2. [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) — Understand structure
3. [ARCHITECTURE.md](#separation-of-concerns)
4. Start contributing!

## 🔗 Key Sections

### Configuration
- Default: PegasusConfig() — Balanced for most use cases
- Memory-optimized: dtype="f16" — 2x smaller
- High-recall: connectivity=64 — Better accuracy
- See [QUICK_START.md](QUICK_START.md) for presets

### Search Modes
- **Vector** — Pure semantic similarity (fast)
- **Keyword** — Full-text search (exact matches)
- **Hybrid** — Combines both with RRF (recommended)

### Components
- EmbeddingProvider — Generate embeddings
- VectorIndexManager — Manage HNSW index
- MetadataStore — SQLite + FTS5 storage
- SearchEngine — Dispatch search modes
- Pegasus — Orchestrate everything

## ✨ Features Matrix

| Feature | Vector | Keyword | Hybrid |
|---------|--------|---------|--------|
| Semantic matching | ✅ | ❌ | ✅ |
| Exact phrases | ❌ | ✅ | ✅ |
| BM25 ranking | ❌ | ✅ | ✅ |
| HNSW speed | ✅ | ❌ | ~ |
| Low latency | ✅ | ✅ | ~ |
| High recall | ~ | ~ | ✅ |

## 🎓 Learning Resources

### Algorithms
- HNSW Paper: https://arxiv.org/abs/1603.09320
- BM25 Scoring: https://en.wikipedia.org/wiki/Okapi_BM25
- RRF Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

### Libraries
- USearch Docs: https://github.com/unum-cloud/usearch
- SQLite FTS5: https://www.sqlite.org/fts5.html
- OpenAI Embeddings API: https://platform.openai.com/docs/api-reference/embeddings

## 🐛 Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| API key not found | Set OPENAI_API_KEY env var |
| Memory usage high | Use dtype="f16" in config |
| Search too slow | Increase expansion_search |
| Import errors | Run pip install -e . |

## 📊 Project Statistics

| Metric | Value |
|--------|-------|
| Package modules | 10 |
| Code lines | 960 |
| Documentation lines | 3000 |
| Public API exports | 11 |
| External dependencies | 9 |
| Python version | 3.8+ |
| License | MIT |

## 🗺️ Architecture at a Glance

```
User Code
    ↓
create_pegasus() or Pegasus(config)
    ↓
┌─ ingest()
│   ├→ load_sources()
│   ├→ chunk_text()
│   ├→ EmbeddingProvider.embed()
│   ├→ MetadataStore.insert_chunk()
│   └→ VectorIndexManager.add()
│
└─ search()
    ├→ EmbeddingProvider.embed()
    ├→ SearchEngine dispatches:
    │   ├→ vector_search()
    │   ├→ keyword_search()
    │   └→ hybrid_search() [RRF]
    └→ SearchResult[]
```

## 📝 Document Purpose Summary

| Doc | Answers | Best For |
|-----|---------|----------|
| README.md | What is Pegasus? | Understanding |
| QUICK_START.md | How do I use it? | Getting started |
| STRUCTURE.md | What's in each module? | Learning code |
| ARCHITECTURE.md | How does it work? | Deep learning |
| PROJECT_MAP.md | Where do I find X? | Navigation |
| REFACTORING_SUMMARY.md | What changed and why? | Context |
| AGENTS.md | How do I contribute? | Development |
| INDEX.md | Where do I start? | Orientation |

## 🎯 Next Steps

1. **First time?** → Read [QUICK_START.md](QUICK_START.md)
2. **Want to learn more?** → Check [STRUCTURE.md](STRUCTURE.md)
3. **Ready to dive deep?** → Read [ARCHITECTURE.md](ARCHITECTURE.md)
4. **Want to contribute?** → Check [AGENTS.md](AGENTS.md)
5. **Lost?** → Use [PROJECT_MAP.md](PROJECT_MAP.md)

---

**Version:** 2.0.0  
**Status:** Production Ready ✅
