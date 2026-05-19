# Pegasus Reliability & Robustness Specification

To run reliably under high load on diverse environments (production web backends, async daemons, desktop applications), Pegasus implements core robustness layers.

## 1. Thread Safety via Reentrant Locks (`RLock`)
Because SQLite and native USearch indices can be read and written concurrently by multiple threads, all entry points into writing and search are protected by an `RLock`:
- Multiple reading threads can access indices safely.
- Ingestion, database modification, and vector insertions are atomic operations under the lock.
- Reentrant locks prevent thread self-deadlocks if a thread makes nested calls to the orchestrator.

## 2. Robust API Retry Policies via Tenacity
Embedding generation relies on external APIs (OpenAI, Jina AI) which are prone to network jitter, rate limits, and short-term outages. All batch embedding retrieval operations are wrapped with `tenacity.retry`:
- **Exponential Backoff**: Starts with 2-second delay up to maximum 10-second delay between attempts.
- **Fail-Safe**: Retries up to 3 times before raising the exception to let the caller handle permanent failure cleanly.
- **In-Memory Query Cache**: If a query has been embedded recently, the cached embedding is used instantly, eliminating external API dependencies and pricing overhead.
