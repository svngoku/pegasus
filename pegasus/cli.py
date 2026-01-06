"""CLI demo for Pegasus RAG engine."""

import json
import os
import sys

from .models import PegasusDoc
from .pegasus import create_pegasus


def demo():
    """Demo usage of Pegasus."""
    print("=== Pegasus v2 Demo ===\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Create engine
    pegasus = create_pegasus(
        db_path="demo.db",
        index_path="demo.usearch",
        dtype="f16",
    )
    
    # Sample documents
    sample_docs = [
        PegasusDoc(
            text="""
            USearch is a high-performance vector search library that implements the HNSW algorithm.
            It supports multiple programming languages including Python, C++, Rust, and JavaScript.
            Key features include SIMD-optimized distance calculations, half-precision support,
            and memory-mapped index serving for production deployments.
            """,
            metadata={"source": "usearch_overview", "title": "USearch Overview"}
        ),
        PegasusDoc(
            text="""
            RAG (Retrieval-Augmented Generation) combines the power of large language models
            with external knowledge retrieval. The process involves: 1) Chunking documents,
            2) Generating embeddings, 3) Storing in a vector database, 4) Retrieving relevant
            chunks for a query, 5) Augmenting the LLM prompt with retrieved context.
            """,
            metadata={"source": "rag_basics", "title": "RAG Fundamentals"}
        ),
        PegasusDoc(
            text="""
            HNSW (Hierarchical Navigable Small World) is an algorithm for approximate nearest
            neighbor search. It builds a multi-layer graph where each layer is a proximity graph
            with decreasing density. Search starts from the top layer and greedily descends,
            providing logarithmic search complexity with high recall.
            """,
            metadata={"source": "hnsw_algorithm", "title": "HNSW Algorithm"}
        ),
    ]
    
    # Ingest documents
    print("Ingesting sample documents...")
    stats = pegasus.ingest(sample_docs, corpus="demo")
    print(f"Ingestion stats: {stats}\n")
    
    # Test vector search
    print("=== Vector Search ===")
    query = "How does HNSW work?"
    results = pegasus.search(query, k=3, mode="vector")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.3f}] {r.content[:100]}...")
    
    print("\n=== Hybrid Search ===")
    query = "vector database performance"
    results = pegasus.search(query, k=3, mode="hybrid")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.3f}] {r.content[:100]}...")
    
    # Show stats
    print(f"\n=== Stats ===")
    print(json.dumps(pegasus.get_stats(), indent=2))
    
    # Cleanup
    pegasus.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    demo()
