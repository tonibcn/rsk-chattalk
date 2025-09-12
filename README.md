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

### 🎯 **Three Modes Available:**

#### 1. **Interactive Chat Mode** 🤖 (Default)
```bash
node rag.js
```
- **Real-time Q&A**: Ask your own questions and get instant answers
- **Chat interface**: Type questions naturally and press Enter
- **Help system**: Type 'help' for example questions
- **Easy exit**: Type 'exit' or 'quit' to end session
- **Clean output**: Perfect for end users

#### 2. **Interactive Debug Mode** 🔍
```bash
DEBUG_MODE=true node rag.js
```
- **Interactive chat** with full visual analytics
- **Real-time debugging**: See how the system processes your questions
- **Visual progress bars** and diagnostic information
- **Perfect for testing and development**

#### 3. **Comprehensive Testing Mode** 🧪
```bash
TEST_MODE=true node rag.js
```
- **Automated testing**: Runs all 20 comprehensive test questions
- **Clean output**: Just the questions and answers
- **Batch processing**: Perfect for validation

#### 4. **Debug Testing Mode** 🔬
```bash
DEBUG_MODE=true TEST_MODE=true node rag.js
```
- **Full diagnostic testing**: All 20 questions with complete analytics
- **Visual analysis**: Every aspect of the RAG pipeline
- **Comprehensive evaluation**: Perfect for deep system analysis

### **Interactive Mode Features:**
**Interactive modes** include:
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

### Example Interactive Session
When running interactive mode, you'll see:

```
═══════════════════════════════════════════════════════════════════════════════
🤖 RSK-CLI INTERACTIVE CHAT
═══════════════════════════════════════════════════════════════════════════════
💬 Ask me anything about rsk-cli!
📝 Type 'help' for example questions
🚪 Type 'exit' or 'quit' to end the session
═══════════════════════════════════════════════════════════════════════════════

❓ Your question: How do I check my wallet balance?

🔄 Processing your question...

💡 Answer: To check your wallet balance using rsk-cli, use the command: `rsk-cli balance`

❓ Your question: help

💡 Example questions you can ask:
  • What is rsk-cli?
  • How do I check my wallet balance?
  • How can I transfer RBTC?
  • What commands are available for smart contracts?
  • How do I create a new wallet?
  • What's the difference between mainnet and testnet?

❓ Your question: exit

👋 Thanks for using rsk-cli chat! Goodbye!
```

### Debug Mode Analytics
When running with `DEBUG_MODE=true`, you get all the visual analytics in real-time during your chat session.

## Troubleshooting
- If embeddings call times out: ensure `nomic-embed-text` is pulled and Ollama is running.
- If answers say “no info,” check the “Top matches” and context preview printed by the script.

## Optional: Persistence (not enabled by default)
- HNSWLib (local, fast, no server)
- Chroma (server via Docker; allows querying/auditing)