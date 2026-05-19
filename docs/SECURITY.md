# Pegasus Security Architecture

Pegasus places top priority on secure, private-by-default execution in any hosting environment.

## 1. Input Sanitization: Preventing FTS5 Query Injection
SQLite FTS5 virtual tables execute query parsing under the hood. Using unsanitized user inputs inside `MATCH` clauses can trigger syntax exceptions or lead to arbitrary full-text searches.

Pegasus enforces strict input escaping via `escape_fts5_query`:
- Double-quotes any raw search query terms.
- Strips potentially hostile syntax characters (such as `*`, `:`, `AND`, `OR`, `NOT`) while preserving semantic meaning.
- Prevents database runtime errors due to malformed FTS syntax strings.

## 2. Secure Secrets & Token Management
All provider tokens (e.g. `OPENAI_API_KEY`, `JINA_API_KEY`, `HF_TOKEN`) are:
- Fetched securely from process environment variables.
- Never logged, cached, or persisted on local SQLite database files or index files.
- Thread-safe and restricted to API runtime instances.

## 3. Sandboxed Paths & Write Permissions
Pegasus prevents file system traversal vulnerabilities:
- Only writes SQLite files (`.db`) and USearch memory-mapped indices (`.usearch`) to explicitly defined local path locations inside the runtime environment.
- Enforces strict folder pathing logic via standard `pathlib.Path` structures.
