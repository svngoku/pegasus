# Pegasus Code Quality & Performance Benchmarks

Pegasus maintains high quality standards across our codebases, APIs, and performance characteristics.

## 1. Quality Gates
We enforce strict linters and test suites to avoid software decay:
- **Linting**: Ruff (checks for code smell, unused imports, formats codebase).
- **Type Checking**: MyPy (maintains static type soundness across our public APIs).
- **Unit Testing**: Pytest (requires comprehensive test coverage for core components, including chunking, config, and caching).

## 2. Performance Metrics
Pegasus target benchmarks are designed around edge devices and local developer setups:
- **Retrieval Latency**: < 2ms per query on typical datasets (10,000+ chunks).
- **Hybrid Fusion Latency**: < 5ms (including RRF ranking calculation on Python list structures).
- **Index Memory Footprint**: ~15MB for 10,000 embeddings (with f16 quantization).
