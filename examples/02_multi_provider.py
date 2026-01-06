#!/usr/bin/env python3
"""
Example 2: Multi-Provider Embeddings with Pegasus

This example demonstrates:
- Using different embedding providers (OpenAI, HuggingFace, Jina)
- Local embeddings with sentence-transformers (free, no API key)
- LLM-based re-ranking for improved relevance
- Switching providers at runtime

Requirements:
    uv add pegasus-rag
    uv add sentence-transformers  # For HuggingFace
    
    # Optional API keys:
    export OPENAI_API_KEY=sk-...      # For OpenAI
    export JINA_API_KEY=jina_...      # For Jina AI
"""

import os
from pathlib import Path

from pegasus import (
    PegasusConfig,
    PegasusDoc,
    Pegasus,
    create_embedding_provider,
    HuggingFaceEmbedding,
    JinaEmbedding,
    rerank_results,
)


def example_huggingface():
    """
    Example using HuggingFace sentence-transformers (local, free).
    No API key required!
    """
    print("\n" + "=" * 60)
    print("🤗 HuggingFace Embeddings (Local, Free)")
    print("=" * 60)
    
    # Create HuggingFace embedding provider
    # Popular models:
    # - all-MiniLM-L6-v2: 384 dims, fast, good quality
    # - all-mpnet-base-v2: 768 dims, better quality
    # - BAAI/bge-small-en-v1.5: 384 dims, excellent quality
    
    print("\n📦 Loading HuggingFace model (first run downloads ~90MB)...")
    
    try:
        embedder = HuggingFaceEmbedding(
            model="all-MiniLM-L6-v2",
            # hf_token="hf_...",  # Optional: for private models
        )
    except ImportError:
        print("❌ sentence-transformers not installed")
        print("   Run: uv add sentence-transformers")
        return
    
    print(f"   Model: {embedder.model}")
    print(f"   Dimension: {embedder.dimension}")
    
    # Configure Pegasus with HuggingFace embeddings
    config = PegasusConfig(
        db_path="example_hf.db",
        index_path="example_hf.usearch",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=embedder.dimension,
        dtype="f32",  # Use f32 for local models
    )
    
    pegasus = Pegasus(config)
    # Replace the default embedder with HuggingFace
    pegasus.embedder = embedder
    pegasus.search_engine.embedder = embedder
    
    # Sample documents
    docs = [
        PegasusDoc(
            text="Python is a high-level programming language known for its readability.",
            metadata={"topic": "programming"}
        ),
        PegasusDoc(
            text="Machine learning enables computers to learn patterns from data.",
            metadata={"topic": "ml"}
        ),
        PegasusDoc(
            text="Neural networks are inspired by biological brain structures.",
            metadata={"topic": "ml"}
        ),
    ]
    
    print("\n📥 Ingesting documents...")
    stats = pegasus.ingest(docs, corpus="demo", show_progress=False)
    print(f"   ✅ Ingested {stats['chunks']} chunks")
    
    print("\n🔍 Searching: 'deep learning models'")
    results = pegasus.search("deep learning models", k=3, mode="vector")
    
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.3f}] {r.content[:60]}...")
    
    pegasus.close()
    
    # Cleanup
    Path("example_hf.db").unlink(missing_ok=True)
    Path("example_hf.usearch").unlink(missing_ok=True)
    
    print("\n✅ HuggingFace example complete!")


def example_provider_factory():
    """
    Example using the provider factory to switch providers easily.
    """
    print("\n" + "=" * 60)
    print("🏭 Provider Factory Pattern")
    print("=" * 60)
    
    providers_to_test = []
    
    # Check which providers are available
    print("\n🔍 Checking available providers...")
    
    # HuggingFace (always available if sentence-transformers installed)
    try:
        hf = create_embedding_provider("huggingface", "all-MiniLM-L6-v2")
        providers_to_test.append(("HuggingFace", hf))
        print("   ✅ HuggingFace: Available (local)")
    except ImportError:
        print("   ⚠️  HuggingFace: Not installed (uv add sentence-transformers)")
    
    # OpenAI (requires API key)
    if os.environ.get("OPENAI_API_KEY"):
        try:
            openai = create_embedding_provider("openai", "text-embedding-3-small")
            providers_to_test.append(("OpenAI", openai))
            print("   ✅ OpenAI: Available (API)")
        except Exception as e:
            print(f"   ⚠️  OpenAI: {e}")
    else:
        print("   ⚠️  OpenAI: No OPENAI_API_KEY set")
    
    # Jina AI (requires API key)
    if os.environ.get("JINA_API_KEY"):
        try:
            jina = create_embedding_provider("jina", "jina-embeddings-v3")
            providers_to_test.append(("Jina", jina))
            print("   ✅ Jina AI: Available (API)")
        except Exception as e:
            print(f"   ⚠️  Jina AI: {e}")
    else:
        print("   ⚠️  Jina AI: No JINA_API_KEY set")
    
    if not providers_to_test:
        print("\n❌ No providers available!")
        return
    
    # Test each provider
    test_texts = ["Hello world", "Machine learning is amazing"]
    
    for name, provider in providers_to_test:
        print(f"\n📊 Testing {name}:")
        print(f"   Model: {provider.model}")
        print(f"   Dimension: {provider.dimension}")
        
        embeddings = provider.embed(test_texts)
        print(f"   Generated {len(embeddings)} embeddings")
        print(f"   Shape: [{len(embeddings)}, {len(embeddings[0])}]")
    
    print("\n✅ Provider factory example complete!")


def example_reranking():
    """
    Example using LLM-based re-ranking to improve search relevance.
    Requires OPENAI_API_KEY.
    """
    print("\n" + "=" * 60)
    print("🎯 LLM Re-ranking for Better Relevance")
    print("=" * 60)
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("\n⚠️  Skipping: OPENAI_API_KEY not set")
        print("   Re-ranking requires an LLM API")
        return
    
    from pegasus import create_pegasus, LLMReranker
    
    print("\n📦 Creating Pegasus instance...")
    pegasus = create_pegasus(
        db_path="example_rerank.db",
        index_path="example_rerank.usearch",
    )
    
    # Documents with varying relevance
    docs = [
        PegasusDoc(
            text="Python is great for web development with frameworks like Django and Flask.",
            metadata={"topic": "web"}
        ),
        PegasusDoc(
            text="Python's scikit-learn library provides machine learning algorithms.",
            metadata={"topic": "ml"}
        ),
        PegasusDoc(
            text="TensorFlow and PyTorch are deep learning frameworks written in Python.",
            metadata={"topic": "dl"}
        ),
        PegasusDoc(
            text="Python can be used for data analysis with pandas and numpy.",
            metadata={"topic": "data"}
        ),
        PegasusDoc(
            text="The Python snake is a non-venomous reptile found in Africa and Asia.",
            metadata={"topic": "animals"}
        ),
    ]
    
    print("\n📥 Ingesting documents...")
    pegasus.ingest(docs, corpus="demo", show_progress=False)
    
    query = "How to build neural networks in Python?"
    
    # Initial search
    print(f"\n🔍 Initial search: '{query}'")
    results = pegasus.search(query, k=5, mode="hybrid")
    
    print("\n   Before re-ranking:")
    for i, r in enumerate(results, 1):
        print(f"   {i}. [{r.score:.3f}] {r.content[:50]}...")
    
    # Re-rank with LLM
    print("\n🤖 Re-ranking with GPT-4o-mini...")
    reranked = rerank_results(
        query=query,
        results=results,
        model="gpt-4o-mini",
        top_n=3,
    )
    
    print("\n   After re-ranking:")
    for i, r in enumerate(reranked, 1):
        llm_score = r.metadata.get("llm_score", "N/A")
        print(f"   {i}. [{r.score:.3f}] (LLM: {llm_score:.2f}) {r.content[:50]}...")
    
    pegasus.close()
    
    # Cleanup
    Path("example_rerank.db").unlink(missing_ok=True)
    Path("example_rerank.usearch").unlink(missing_ok=True)
    
    print("\n✅ Re-ranking example complete!")


def main():
    print("=" * 60)
    print("Example 2: Multi-Provider Embeddings")
    print("=" * 60)
    
    # Example 1: HuggingFace (local, free)
    example_huggingface()
    
    # Example 2: Provider factory pattern
    example_provider_factory()
    
    # Example 3: LLM re-ranking
    example_reranking()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
