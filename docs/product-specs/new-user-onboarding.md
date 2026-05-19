# Pegasus New User Onboarding Spec

This spec outlines the onboarding flow and user experience for developer onboarding to the Pegasus RAG engine.

## 1. Quick Setup & Environment Setup
To minimize friction, the setup flow must require exactly three steps:
1. **Installation**: Install the package and dependencies via modern package managers:
   ```bash
   pip install pegasus-rag
   ```
2. **Configuration**: Provide an API key as an environment variable:
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```
3. **Execution**: Initialize the RAG instance and run a query in under 5 lines of code:
   ```python
   from pegasus import create_pegasus
   db = create_pegasus("my.db", "my.usearch")
   db.ingest(["Pegasus makes building high-performance RAG fast and simple."])
   print(db.search("Is Pegasus fast?", mode="hybrid")[0].content)
   ```

## 2. Diagnostics (Self-Assessment)
Onboarding users often hit integration blocks (e.g., missing API keys, optional packages not installed). To guide them, Pegasus exposes `check_installation()`:
- Verifies package versions.
- Checks local/remote providers available (`openai`, `huggingface`, `jina`).
- Prints diagnostic checks for system paths and write permissions.
