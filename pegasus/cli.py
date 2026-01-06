"""CLI for Pegasus RAG engine."""

import argparse
import json
import os
import sys

# Set USER_AGENT to suppress langchain warning
if not os.environ.get("USER_AGENT"):
    os.environ["USER_AGENT"] = "pegasus-rag/2.1.0"


def demo(args: argparse.Namespace) -> None:
    """Run demo with sample documents."""
    from .models import PegasusDoc
    from .pegasus import create_pegasus
    
    print("=== Pegasus v2 Demo ===\n")
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        print("  export OPENAI_API_KEY=sk-...")
        sys.exit(1)
    
    # Create engine
    pegasus = create_pegasus(
        db_path=args.db or "demo.db",
        index_path=args.index or "demo.usearch",
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


def serve(args: argparse.Namespace) -> None:
    """Start the REST API server."""
    try:
        import uvicorn
        from .api import create_app
    except ImportError:
        print("Error: API dependencies not installed.")
        print("  Run: uv sync --extra api")
        sys.exit(1)
    
    app = create_app(
        db_path=args.db or "pegasus.db",
        index_path=args.index or "pegasus.usearch",
    )
    
    print(f"Starting Pegasus API server on http://{args.host}:{args.port}")
    print(f"  Database: {args.db or 'pegasus.db'}")
    print(f"  Index: {args.index or 'pegasus.usearch'}")
    print(f"  Docs: http://{args.host}:{args.port}/docs\n")
    
    uvicorn.run(app, host=args.host, port=args.port)


def stats(args: argparse.Namespace) -> None:
    """Show engine statistics."""
    from .pegasus import create_pegasus
    
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: Set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    pegasus = create_pegasus(
        db_path=args.db or "pegasus.db",
        index_path=args.index or "pegasus.usearch",
    )
    
    info = pegasus.get_stats()
    print(json.dumps(info, indent=2))
    pegasus.close()


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="pegasus",
        description="Pegasus - High-Performance RAG Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pegasus demo              Run demo with sample documents
  pegasus serve             Start REST API server
  pegasus stats             Show engine statistics

Environment variables:
  OPENAI_API_KEY    Required for OpenAI embeddings
  HF_TOKEN          Optional for HuggingFace models
  JINA_API_KEY      Required for Jina AI embeddings
"""
    )
    
    parser.add_argument(
        "--version", action="version", version="%(prog)s 2.1.0"
    )
    parser.add_argument(
        "--db", type=str, help="Database path (default: pegasus.db)"
    )
    parser.add_argument(
        "--index", type=str, help="Index path (default: pegasus.usearch)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Demo command
    demo_parser = subparsers.add_parser("demo", help="Run demo with sample documents")
    demo_parser.set_defaults(func=demo)
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start REST API server")
    serve_parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Host to bind (default: 127.0.0.1)"
    )
    serve_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind (default: 8000)"
    )
    serve_parser.set_defaults(func=serve)
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show engine statistics")
    stats_parser.set_defaults(func=stats)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    
    args.func(args)


if __name__ == "__main__":
    main()
