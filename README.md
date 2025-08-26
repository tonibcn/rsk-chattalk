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

### Debug Mode
For detailed debugging information, you can enable debug mode:
```bash
DEBUG_MODE=true node rag.js
```

**Normal mode** shows:
- Clean, minimal output
- Final answers only
- 2 sample questions

**Debug mode** shows:
- 📊 **Visual Progress Bars**: Similarity scores, context length, response length
- 📁 **File Loading Analytics**: Character counts, file statistics, samples
- 📚 **Document Processing Pipeline**: Type breakdown, chunk statistics
- 🧠 **Embedding Statistics**: Vector dimensions, memory usage estimates
- 🔍 **Similarity Analysis**: Top matches with score bars and previews
- 🎯 **Keyword Matching**: Context analysis with match rate percentages
- 📤 **Context Preview**: Shows exact text sent to the model
- 🤖 **Response Analysis**: Quality checks, source attribution validation
- 🧪 **Comprehensive Testing**: All 20 questions with progress tracking
- 💡 **Special Debugging**: Enhanced analysis for specific question types
- ✅ **Visual Status Indicators**: Color-coded success/warning messages

### Example Debug Output
When running with `DEBUG_MODE=true`, you'll see detailed visualizations like:

```
═══════════════════════════════════════════════════════════════════════════════
📊 TOP SIMILARITY MATCHES
═══════════════════════════════════════════════════════════════════════════════

🔍 MATCH #1
   📈 Score: 0.7845 [████████████████░░░░]
   📁 Source: README.md
   📝 Type: documentation
   💬 Preview: "rsk-cli is a command-line tool for interacting with Rootstock blockchain..."
   ────────────────────────────────────────────────────────────────

🎯 KEYWORD ANALYSIS:
   🔎 Searching for: [wallet, balance, command]
   ✅ Found: [wallet, balance, command]
   📊 Match rate: 3/3 (100%)

🤖 MODEL RESPONSE ANALYSIS
═══════════════════════════════════════════════════════════════════════════════
To check your wallet balance using rsk-cli, use the command: `rsk-cli balance`
═══════════════════════════════════════════════════════════════════════════════
📏 Response length: 87 chars [████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]
✅ 🟢 Model successfully used the provided context
📚 Source attribution: ✅
```

## Troubleshooting
- If embeddings call times out: ensure `nomic-embed-text` is pulled and Ollama is running.
- If answers say “no info,” check the “Top matches” and context preview printed by the script.

## Optional: Persistence (not enabled by default)
- HNSWLib (local, fast, no server)
- Chroma (server via Docker; allows querying/auditing)