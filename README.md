# rsk-chattalk (RAG over rsk-cli docs + source)

This project demonstrates a simple RAG pipeline using Ollama + LangChain that:
- Reads `docs/README.md` (and optionally `src/` from rsk-cli)
- Chunks and embeds content with `nomic-embed-text` via Ollama
- Retrieves top chunks and answers with `llama3.2`

## Prerequisites
- Node.js 18+
- Ollama installed and running
- Models pulled:
  - `ollama pull llama3.2`
  - `ollama pull nomic-embed-text`

## Install
```bash
npm install
```

## Data sources
- Required: `docs/README.md`
- Optional: `src/` folder from rsk-cli to increase accuracy (already included here for testing).
  - Source: https://github.com/rsksmart/rsk-cli/tree/main/src
  - Note: Keep upstream license/attribution if you include files.

## Start Ollama (required)
Ollama must be running and the models available.

- Start server (one of):
  - Open the Ollama app, or
  - Terminal:
    ```bash
    ollama serve
    ```
- Verify and pull models:
  ```bash
  curl http://localhost:11434/api/tags
  ollama pull nomic-embed-text
  ollama pull llama3.2
  ```

If localhost fails on Windows, try:
- Use baseUrl `http://127.0.0.1:11434` in `rag.js`
- Check firewall/antivirus for port 11434

## Run
```bash
node rag.js
```
You’ll see:
- How many files/chunks were loaded
- Top matches (scores and file paths)
- The exact context fed to the model
- Answers to multiple test questions

## Troubleshooting
- If embeddings call times out: ensure `nomic-embed-text` is pulled and Ollama is running.
- If answers say “no info,” check the “Top matches” and context preview printed by the script.

## Optional: Persistence (not enabled by default)
- HNSWLib (local, fast, no server)
- Chroma (server via Docker; allows querying/auditing)