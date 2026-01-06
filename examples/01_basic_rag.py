#!/usr/bin/env python3
"""
Example 1: Basic RAG Pipeline with Pegasus

This example demonstrates:
- Creating a Pegasus instance
- Ingesting documents from multiple sources
- Searching with different modes (vector, keyword, hybrid)
- Using progress callbacks
- Exporting/importing corpora

Requirements:
    uv add pegasus-rag  # or: pip install pegasus-rag
    export OPENAI_API_KEY=sk-...
"""

import os
from pathlib import Path

# Ensure API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  Set OPENAI_API_KEY environment variable")
    print("   export OPENAI_API_KEY=sk-...")
    exit(1)

from pegasus import (
    Pegasus,
    PegasusConfig,
    PegasusDoc,
    create_pegasus,
    load_sources,
    get_cache,
)


def main():
    print("=" * 60)
    print("Example 1: Basic RAG Pipeline")
    print("=" * 60)
    
    # ============================================================
    # Step 1: Create Pegasus instance
    # ============================================================
    print("\n📦 Creating Pegasus instance...")
    
    # Option A: Quick setup with defaults
    pegasus = create_pegasus(
        db_path="example_basic.db",
        index_path="example_basic.usearch",
    )
    
    # Option B: Custom configuration
    # config = PegasusConfig(
    #     db_path="example.db",
    #     index_path="example.usearch",
    #     embedding_model="text-embedding-3-small",  # Cheaper model
    #     embedding_dim=1536,
    #     dtype="f16",  # Half precision for 2x memory savings
    #     chunk_size=256,  # Smaller chunks
    #     chunk_overlap=32,
    #     hybrid_alpha=0.7,  # 70% vector, 30% keyword
    # )
    # pegasus = Pegasus(config)
    
    # ============================================================
    # Step 2: Prepare documents
    # ============================================================
    print("\n📄 Preparing documents...")
    
    # Create documents programmatically
    documents = [
        PegasusDoc(
            text="""
            Vector databases are specialized systems designed to store and query 
            high-dimensional vectors efficiently. Unlike traditional databases that 
            use exact matching, vector databases use approximate nearest neighbor (ANN) 
            algorithms to find similar items based on their vector representations.
            
            Popular vector databases include Pinecone, Weaviate, Milvus, and Qdrant.
            Each offers different trade-offs between performance, scalability, and features.
            """,
            metadata={
                "source": "knowledge_base",
                "title": "Introduction to Vector Databases",
                "category": "databases",
            }
        ),
        PegasusDoc(
            text="""
            RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses 
            by retrieving relevant context from external knowledge sources. The process involves:
            
            1. Document Chunking: Split documents into manageable pieces
            2. Embedding Generation: Convert text chunks to vector representations
            3. Vector Storage: Store embeddings in a vector database
            4. Query Processing: Convert user query to embedding
            5. Retrieval: Find most similar chunks to the query
            6. Augmentation: Add retrieved context to LLM prompt
            7. Generation: LLM generates response with context
            """,
            metadata={
                "source": "knowledge_base", 
                "title": "RAG Architecture Overview",
                "category": "architecture",
            }
        ),
        PegasusDoc(
            text="""
            HNSW (Hierarchical Navigable Small World) is an algorithm for approximate 
            nearest neighbor search. It constructs a multi-layer graph where:
            
            - The bottom layer contains all vectors
            - Higher layers contain progressively fewer vectors (subset)
            - Search starts from top layer and descends
            - Each layer uses greedy search to find closest neighbors
            
            Key parameters:
            - M (connectivity): Number of bi-directional links per node
            - efConstruction: Size of dynamic candidate list during index building
            - ef: Size of dynamic candidate list during search
            
            HNSW provides O(log n) search complexity with high recall rates.
            """,
            metadata={
                "source": "knowledge_base",
                "title": "HNSW Algorithm Deep Dive",
                "category": "algorithms",
            }
        ),
    ]
    
    print(f"   Prepared {len(documents)} documents")
    
    # ============================================================
    # Step 3: Ingest documents with progress tracking
    # ============================================================
    print("\n📥 Ingesting documents...")
    
    # Progress callback
    def on_progress(info):
        pct = (info["doc_index"] / info["total_docs"]) * 100
        print(f"   Progress: {info['doc_index']}/{info['total_docs']} ({pct:.0f}%)")
    
    stats = pegasus.ingest(
        documents,
        corpus="knowledge_base",
        show_progress=False,
        on_progress=on_progress,
    )
    
    print(f"   ✅ Ingested {stats['chunks']} chunks, skipped {stats['skipped']} duplicates")
    
    # ============================================================
    # Step 4: Search examples
    # ============================================================
    
    # --- Vector Search ---
    print("\n🔍 Vector Search: 'How does HNSW work?'")
    results = pegasus.search(
        "How does HNSW work?",
        k=3,
        mode="vector",
        corpus="knowledge_base",
    )
    
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.3f}] {r.metadata.get('title', 'N/A')}")
        print(f"      {r.content[:100].strip()}...")
    
    # --- Keyword Search ---
    print("\n🔍 Keyword Search: 'vector database'")
    results = pegasus.search(
        "vector database",
        k=3,
        mode="keyword",
        corpus="knowledge_base",
    )
    
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.3f}] {r.metadata.get('title', 'N/A')}")
    
    # --- Hybrid Search (recommended) ---
    print("\n🔍 Hybrid Search: 'explain RAG retrieval process'")
    results = pegasus.search(
        "explain RAG retrieval process",
        k=3,
        mode="hybrid",
        hybrid_alpha=0.7,  # 70% vector, 30% keyword
    )
    
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.3f}] {r.metadata.get('title', 'N/A')}")
    
    # ============================================================
    # Step 5: Show statistics
    # ============================================================
    print("\n📊 Engine Statistics:")
    stats = pegasus.get_stats()
    print(f"   Index size: {stats['index_size']} vectors")
    print(f"   Corpora: {len(stats['corpora'])}")
    for corpus in stats["corpora"]:
        print(f"     - {corpus['corpus']}: {corpus['chunks']} chunks, {corpus['docs']} docs")
    
    # Cache stats
    cache_stats = get_cache().stats()
    print(f"   Cache: {cache_stats['hits']} hits, {cache_stats['misses']} misses")
    print(f"   Hit rate: {cache_stats['hit_rate']:.1%}")
    
    # ============================================================
    # Step 6: Export corpus (backup)
    # ============================================================
    print("\n💾 Exporting corpus to backup...")
    export_path = "knowledge_base_backup.jsonl"
    count = pegasus.export_corpus("knowledge_base", export_path)
    print(f"   ✅ Exported {count} chunks to {export_path}")
    
    # ============================================================
    # Cleanup
    # ============================================================
    pegasus.close()
    
    # Clean up example files
    for f in ["example_basic.db", "example_basic.usearch", export_path]:
        Path(f).unlink(missing_ok=True)
    
    print("\n✅ Example complete!")


if __name__ == "__main__":
    main()
