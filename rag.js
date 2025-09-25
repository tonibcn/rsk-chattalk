// rag.js
import fs from "fs";
import { Ollama } from "@langchain/ollama";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";
import readline from "readline";
import crypto from "crypto";
import FAQSystem from "./faq-system.js";

// 0. Mode configuration
const DEBUG_MODE = process.env.DEBUG_MODE === "true";   // Show analytics during interaction
const TEST_MODE = process.env.TEST_MODE === "true";     // Run all 20 test questions
const INTERACTIVE = !TEST_MODE;                         // Interactive chat by default

// Performance optimizations
const CACHE_EMBEDDINGS = true;                          // Cache embeddings to avoid re-computation
const FAST_MODE = process.env.FAST_MODE === "true";     // Use optimizations for speed
const ULTRA_FAST = process.env.ULTRA_FAST === "true";   // Maximum speed optimizations
const CACHE_FILE = "./embeddings-cache.json";           // Persistent cache file
const embeddingCache = new Map();                       // In-memory cache for embeddings

// FAQ System for instant responses
const faqSystem = new FAQSystem();

// Load persistent cache on startup
function loadPersistentCache() {
  try {
    if (fs.existsSync(CACHE_FILE)) {
      const cacheData = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
      Object.entries(cacheData).forEach(([key, value]) => {
        embeddingCache.set(key, value);
      });
      if (DEBUG_MODE) {
        console.log(`🚀 Loaded ${Object.keys(cacheData).length} cached embeddings from disk`);
      }
      return Object.keys(cacheData).length;
    }
  } catch (error) {
    if (DEBUG_MODE) {
      console.log(`⚠️ Could not load cache file: ${error.message}`);
    }
  }
  return 0;
}

// Save cache to disk
function savePersistentCache() {
  try {
    const cacheData = Object.fromEntries(embeddingCache.entries());
    fs.writeFileSync(CACHE_FILE, JSON.stringify(cacheData, null, 2));
    if (DEBUG_MODE) {
      console.log(`💾 Saved ${Object.keys(cacheData).length} embeddings to persistent cache`);
    }
  } catch (error) {
    if (DEBUG_MODE) {
      console.log(`⚠️ Could not save cache file: ${error.message}`);
    }
  }
}

// 1. Ollama configuration with timeouts
const ollamaModel = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3.2",
  timeout: ULTRA_FAST ? 10000 : (FAST_MODE ? 20000 : 60000), // Ultra fast: 10s, Fast: 20s, Normal: 60s
});

// Utility: cosine similarity
function cosineSimilarity(a, b) {
  let dot = 0;
  let aNorm = 0;
  let bNorm = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i += 1) {
    const av = a[i];
    const bv = b[i];
    dot += av * bv; // Fixed: was av * av, should be av * bv
    aNorm += av * av;
    bNorm += bv * bv;
  }
  if (aNorm === 0 || bNorm === 0) return 0;
  return dot / (Math.sqrt(aNorm) * Math.sqrt(bNorm));
}

// Function to read source files
function readCommandsFolder() {
  const srcPath = "./src";
  const documents = [];
  
  try {
    if (fs.existsSync(srcPath)) {
      if (DEBUG_MODE) console.log("📁 Reading source files (src)...");
      const files = fs.readdirSync(srcPath, { recursive: true });
      
      files.forEach(file => {
        if (file.endsWith('.js') || file.endsWith('.ts')) {
          const filePath = `${srcPath}/${file}`;
          try {
            const content = fs.readFileSync(filePath, "utf8");
            // Filter out very short or empty files
            if (content.length > 100) {
              documents.push(new Document({ 
                pageContent: content,
                metadata: { source: filePath, type: "source_code" }
              }));
              if (DEBUG_MODE) console.log(`  ✅ ${file} (${content.length} chars)`);
            } else {
              if (DEBUG_MODE) console.log(`  ⚠️  ${file} (too short: ${content.length} chars)`);
            }
          } catch (err) {
            if (DEBUG_MODE) console.log(`  ❌ Error reading ${file}: ${err.message}`);
          }
        }
      });
      
      if (DEBUG_MODE) {
        console.log(`📁 Total source files read: ${documents.length}`);
        console.log("┌" + "─".repeat(70) + "┐");
        console.log("│ 📊 SOURCE FILES SUMMARY                                          │");
        console.log("└" + "─".repeat(70) + "┘");
        
        // Show file statistics
        if (documents.length > 0) {
          const sizes = documents.map(d => d.pageContent.length);
          const totalSize = sizes.reduce((a, b) => a + b, 0);
          const avgSize = Math.round(totalSize / sizes.length);
          
          console.log(`   📈 Total characters: ${totalSize.toLocaleString()}`);
          console.log(`   📊 Average file size: ${avgSize.toLocaleString()} chars`);
          console.log(`   📁 Largest file: ${Math.max(...sizes).toLocaleString()} chars`);
          console.log(`   📄 Smallest file: ${Math.min(...sizes).toLocaleString()} chars`);
          
          console.log("\n📝 Sample from first file:");
          const sample = documents[0].pageContent.substring(0, 200);
          console.log(`   "${sample}..."`);
        }
      }
    } else {
      if (DEBUG_MODE) console.log("⚠️ 'src' folder not found. Only README.md will be used");
    }
  } catch (error) {
    if (DEBUG_MODE) console.log(`⚠️  Error reading source folder: ${error.message}`);
  }
  
  return documents;
}

// 2. Load multiple information sources
try {
  // Load persistent cache first
  const startupTime = Date.now();
  const cachedCount = loadPersistentCache();
  
  // Initialize FAQ system
  const faqStats = faqSystem.getStats();
  if (DEBUG_MODE) {
    console.log(`📚 FAQ System loaded: ${faqStats.questions} questions across ${faqStats.categories} categories`);
    console.log(`⚙️ Executable commands: ${faqStats.commands}`);
  }
  
  const allDocs = [];
  
  // Read README.md
  if (DEBUG_MODE) console.log("📖 Reading README.md...");
const readmeContent = fs.readFileSync("./docs/README.md", "utf8");
  allDocs.push(new Document({ 
    pageContent: readmeContent,
    metadata: { source: "README.md", type: "documentation" }
  }));
  if (DEBUG_MODE) console.log("✅ README.md read successfully, size:", readmeContent.length, "characters");
  
  // Read source files
  const commandDocs = readCommandsFolder();
  allDocs.push(...commandDocs);
  
  if (DEBUG_MODE) {
    console.log("\n" + "═".repeat(80));
    console.log("📚 DOCUMENT PROCESSING PIPELINE");
    console.log("═".repeat(80));
    console.log(`📁 Total documents loaded: ${allDocs.length}`);
    
    // Show document breakdown
    const docTypes = {};
    allDocs.forEach(doc => {
      docTypes[doc.metadata.type] = (docTypes[doc.metadata.type] || 0) + 1;
    });
    
    console.log("📊 Document types:");
    Object.entries(docTypes).forEach(([type, count]) => {
      const bar = "█".repeat(Math.round(count * 5)) + "░".repeat(Math.max(10 - Math.round(count * 5), 0));
      console.log(`   ${type}: ${count} [${bar}]`);
    });
  }
  
const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: ULTRA_FAST ? 500 : (FAST_MODE ? 800 : 1000), // Ultra fast: 500, Fast: 800, Normal: 1000
    chunkOverlap: ULTRA_FAST ? 50 : (FAST_MODE ? 80 : 100), // Ultra fast: 50, Fast: 80, Normal: 100
  });

  const docs = await splitter.splitDocuments(allDocs);
  
  if (DEBUG_MODE) {
    console.log(`\n🔪 Text chunking complete: ${docs.length} chunks created`);
    
    // Show chunk statistics
    const chunkSizes = docs.map(d => d.pageContent.length);
    const avgChunkSize = Math.round(chunkSizes.reduce((a, b) => a + b, 0) / chunkSizes.length);
    const chunkBar = "█".repeat(Math.round(avgChunkSize / 50)) + "░".repeat(Math.max(20 - Math.round(avgChunkSize / 50), 0));
    
    console.log(`   📊 Average chunk size: ${avgChunkSize} chars [${chunkBar}]`);
    console.log(`   📏 Size range: ${Math.min(...chunkSizes)} - ${Math.max(...chunkSizes)} chars`);
  }

  // 3. Create embeddings (with caching for performance)
  const startTime = Date.now();
  if (DEBUG_MODE) {
    console.log("\n🔄 Creating embeddings with 'nomic-embed-text'...");
    console.log("   ⏳ This may take a moment depending on document count...");
    if (CACHE_EMBEDDINGS) {
      console.log("   🚀 Embedding caching enabled for better performance");
    }
  }
  
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
    timeout: ULTRA_FAST ? 8000 : (FAST_MODE ? 15000 : 30000), // Ultra fast: 8s, Fast: 15s, Normal: 30s
  });

  const docTexts = docs.map(d => d.pageContent);
  let docVectors;
  
  if (CACHE_EMBEDDINGS) {
    // Generate cache key based on document content
    const contentHash = crypto.createHash('md5').update(JSON.stringify(docTexts)).digest('hex');
    const cacheKey = `embeddings_${contentHash}`;
    
    if (embeddingCache.has(cacheKey)) {
      if (DEBUG_MODE) console.log("   ⚡ Using cached embeddings!");
      docVectors = embeddingCache.get(cacheKey);
    } else {
      if (DEBUG_MODE) console.log("   🔄 Computing new embeddings...");
      docVectors = await embeddings.embedDocuments(docTexts);
      embeddingCache.set(cacheKey, docVectors);
      // Save to persistent cache immediately
      savePersistentCache();
      if (DEBUG_MODE) console.log("   💾 Embeddings cached for future use");
    }
  } else {
    docVectors = await embeddings.embedDocuments(docTexts);
  }
  
  const index = docVectors.map((vec, i) => ({ 
    vector: vec, 
    text: docTexts[i],
    metadata: docs[i].metadata 
  }));
  
  const embeddingTime = Date.now() - startTime;
  
  if (DEBUG_MODE) {
    console.log("✅ Embeddings created successfully!");
    console.log("┌" + "─".repeat(50) + "┐");
    console.log("│ 🧠 EMBEDDING STATISTICS                        │");
    console.log("└" + "─".repeat(50) + "┘");
    console.log(`   📊 Total vectors: ${index.length}`);
    console.log(`   🔢 Vector dimensions: ${docVectors[0]?.length || 0}`);
    console.log(`   💾 Memory usage: ~${Math.round(index.length * (docVectors[0]?.length || 0) * 4 / 1024 / 1024)} MB`);
    console.log(`   ⚡ Processing time: ${(embeddingTime / 1000).toFixed(2)}s`);
    console.log(`   📁 Cached embeddings loaded: ${cachedCount}`);
    console.log(`   🚀 Total startup time: ${((Date.now() - startupTime) / 1000).toFixed(2)}s`);
    if (FAST_MODE || ULTRA_FAST) {
      const mode = ULTRA_FAST ? "ULTRA_FAST" : "FAST";
      console.log(`   🚀 ${mode} mode optimizations: ENABLED`);
    }
  }

  // 4. Ask function
async function askQuestion(question) {
    try {
      const questionStartTime = Date.now();
      if (DEBUG_MODE) console.log("\n❓ Question:", question);
      
      // 🚀 STEP 1: Check FAQ first for instant responses
      if (DEBUG_MODE) console.log("📚 Checking FAQ system...");
      const faqResponse = await faqSystem.getInstantAnswer(question);
      
      if (faqResponse && faqResponse.confidence > 0.6) {
        const faqTime = Date.now() - questionStartTime;
        
        if (DEBUG_MODE) {
          console.log("✅ FAQ Match Found!");
          console.log(`   📊 Confidence: ${(faqResponse.confidence * 100).toFixed(1)}%`);
          console.log(`   📁 Category: ${faqResponse.category}`);
          console.log(`   🎯 Matched: "${faqResponse.matchedQuestion}"`);
          if (faqResponse.command) {
            console.log(`   ⚙️ Command: ${faqResponse.command}`);
          }
        }
        
        const timeIcon = ULTRA_FAST ? "🚀" : (FAST_MODE ? "⚡" : "⏱️");
        console.log(`${timeIcon} Response time: ${(faqTime / 1000).toFixed(2)}s (FAQ)`);
        console.log("\n💡 Answer:", faqResponse.answer);
        
        if (faqResponse.command && faqResponse.executable) {
          console.log(`\n🔧 **Command**: \`${faqResponse.command}\``);
        }
        
        return;
      }
      
      // 🔄 STEP 2: Fallback to RAG if no FAQ match
      if (DEBUG_MODE) {
        if (faqResponse) {
          console.log(`⚠️ FAQ confidence too low (${(faqResponse.confidence * 100).toFixed(1)}%), using RAG`);
        } else {
          console.log("❌ No FAQ match found, using RAG");
        }
        console.log("🔍 Computing query embedding...");
      }
      
      // Cache query embeddings too
      let queryVec;
      if (CACHE_EMBEDDINGS) {
        const queryHash = crypto.createHash('md5').update(question).digest('hex');
        const queryCacheKey = `query_${queryHash}`;
        
        if (embeddingCache.has(queryCacheKey)) {
          if (DEBUG_MODE) console.log("   ⚡ Using cached query embedding!");
          queryVec = embeddingCache.get(queryCacheKey);
        } else {
          queryVec = await embeddings.embedQuery(question);
          embeddingCache.set(queryCacheKey, queryVec);
          // Save query cache to persistent storage
          savePersistentCache();
        }
      } else {
        queryVec = await embeddings.embedQuery(question);
      }

      if (DEBUG_MODE) console.log("🔎 Searching similar documents...");
      
      // Optimized similarity search with early termination for fast mode
      let scored;
      if (FAST_MODE || ULTRA_FAST) {
        // Use a more efficient search for speed
        scored = [];
        const targetCount = ULTRA_FAST ? 2 : 3;
        let minScore = 0;
        
        for (const { vector, text, metadata } of index) {
          const score = cosineSimilarity(queryVec, vector);
          
          if (scored.length < targetCount) {
            scored.push({ score, text, metadata });
            if (scored.length === targetCount) {
              scored.sort((a, b) => b.score - a.score);
              minScore = scored[scored.length - 1].score;
            }
          } else if (score > minScore) {
            scored[scored.length - 1] = { score, text, metadata };
            scored.sort((a, b) => b.score - a.score);
            minScore = scored[scored.length - 1].score;
          }
        }
      } else {
        // Full search for accuracy
        scored = index.map(({ vector, text, metadata }) => ({
          score: cosineSimilarity(queryVec, vector),
          text,
          metadata
        }));
        scored.sort((a, b) => b.score - a.score);
      }
      const topK = scored.slice(0, ULTRA_FAST ? 2 : (FAST_MODE ? 3 : 5)); // Ultra fast: 2, Fast: 3, Normal: 5

      if (DEBUG_MODE) {
        console.log("\n" + "═".repeat(80));
        console.log("📊 TOP SIMILARITY MATCHES");
        console.log("═".repeat(80));
        topK.forEach((item, i) => {
          const scoreBar = "█".repeat(Math.round(item.score * 20)) + "░".repeat(20 - Math.round(item.score * 20));
          console.log(`\n🔍 MATCH #${i + 1}`);
          console.log(`   📈 Score: ${item.score.toFixed(4)} [${scoreBar}]`);
          console.log(`   📁 Source: ${item.metadata.source}`);
          console.log(`   📝 Type: ${item.metadata.type}`);
          console.log(`   💬 Preview: "${item.text.substring(0, 120)}..."`);
          console.log("   " + "─".repeat(60));
        });
      }

      const context = topK.map(r => r.text).join("\n\n---\n\n");
      
      if (DEBUG_MODE) {
        // Verify that the context contains useful information
        console.log("\n" + "═".repeat(80));
        console.log("🔍 CONTEXT ANALYSIS");
        console.log("═".repeat(80));
        
        const contextLength = context.length;
        const lengthBar = "█".repeat(Math.min(Math.round(contextLength / 100), 40)) + "░".repeat(Math.max(40 - Math.round(contextLength / 100), 0));
        console.log(`📏 Context length: ${contextLength} chars [${lengthBar}]`);
        
        // Keyword scan
        const keywords = question.toLowerCase().split(' ').filter(word => word.length > 3);
        const contextLower = context.toLowerCase();
        const foundKeywords = keywords.filter(keyword => contextLower.includes(keyword));
        
        console.log(`\n🎯 KEYWORD ANALYSIS:`);
        console.log(`   🔎 Searching for: [${keywords.join(', ')}]`);
        console.log(`   ✅ Found: [${foundKeywords.join(', ')}]`);
        console.log(`   📊 Match rate: ${foundKeywords.length}/${keywords.length} (${Math.round(foundKeywords.length/keywords.length*100)}%)`);
        
        // Special debugging for balance-related questions
        if (question.toLowerCase().includes('balance')) {
          console.log("\n" + "┌" + "─".repeat(60) + "┐");
          console.log("│ 🔍 SPECIAL: Balance Question Analysis               │");
          console.log("└" + "─".repeat(60) + "┘");
          
          const balanceContent = contextLower.includes('balance') ? '✅' : '❌';
          const commandContent = contextLower.includes('command') ? '✅' : '❌';
          const walletContent = contextLower.includes('wallet') ? '✅' : '❌';
          
          console.log(`   💰 Balance terms: ${balanceContent}`);
          console.log(`   ⚙️  Command terms: ${commandContent}`);
          console.log(`   👛 Wallet terms: ${walletContent}`);
          
          // Show specific balance-related snippets
          const balanceSnippets = context.split('\n').filter(line => 
            line.toLowerCase().includes('balance') || 
            line.toLowerCase().includes('command') ||
            line.toLowerCase().includes('wallet')
          );
          
          if (balanceSnippets.length > 0) {
            console.log(`\n   📝 Relevant snippets (${balanceSnippets.length} found):`);
            balanceSnippets.slice(0, 3).forEach((snippet, i) => {
              console.log(`   ${i + 1}. "${snippet.trim().substring(0, 80)}..."`);
            });
          }
        }
        
        if (foundKeywords.length === 0) {
          console.log("\n⚠️  WARNING: No relevant keywords found in context");
        }
        
        // Show a preview of the context being sent to the model
        console.log("\n" + "═".repeat(80));
        console.log("📤 CONTEXT PREVIEW (First 300 chars)");
        console.log("═".repeat(80));
        console.log(`"${context.substring(0, 300)}..."`);
        console.log("═".repeat(80));
      }

  let prompt;
  if (ULTRA_FAST) {
    // Ultra-short prompt for maximum speed
    prompt = `Context: ${context}

Question: ${question}

Answer briefly using only the context above:`;
  } else if (FAST_MODE) {
    // Medium prompt for balance of speed and quality
    prompt = `You are an rsk-cli expert. Answer using ONLY the provided context.

Context: ${context}

Question: ${question}

Answer:`;
  } else {
    // Full detailed prompt for accuracy
    prompt = `
You are an rsk-cli expert. Answer the following question using ONLY the information from the provided context.

IMPORTANT:
- If you find relevant information in the context, use it to answer in detail.
- If the information is not in the context, reply "I don't have that information in the provided context".
- Explicitly mention which parts of the context you used for your answer.

Context:
${context}

Question: ${question}

Instructions:
1. Carefully analyze the context
2. Identify information relevant to the question
3. Provide a detailed answer based on that information
4. Mention specifically which files or sections of the context you used

Answer:
`;
  }

      if (DEBUG_MODE) console.log("🤖 Generating answer...");
      
      // Show processing time in debug mode
      const processingTime = Date.now() - questionStartTime;
      if (DEBUG_MODE) {
        console.log(`⚡ Query processing time: ${(processingTime / 1000).toFixed(2)}s`);
      }

  const response = await ollamaModel.call(prompt);
      
      const totalTime = Date.now() - questionStartTime;
      if (DEBUG_MODE) {
        console.log(`🏁 Total response time: ${(totalTime / 1000).toFixed(2)}s`);
      } else {
        // Always show response time in non-debug mode for performance comparison
        const timeIcon = ULTRA_FAST ? "🚀" : (FAST_MODE ? "⚡" : "⏱️");
        console.log(`${timeIcon} Response time: ${(totalTime / 1000).toFixed(2)}s`);
      }
      
      if (DEBUG_MODE) {
        console.log("\n" + "═".repeat(80));
        console.log("🤖 MODEL RESPONSE ANALYSIS");
        console.log("═".repeat(80));
        console.log(response);
        console.log("═".repeat(80));
        
        // Verificar si la respuesta menciona el contexto
        const responseLength = response.length;
        const responseBar = "█".repeat(Math.min(Math.round(responseLength / 20), 40)) + "░".repeat(Math.max(40 - Math.round(responseLength / 20), 0));
        console.log(`📏 Response length: ${responseLength} chars [${responseBar}]`);
        
        if (
          response.toLowerCase().includes("i don't have") ||
          response.toLowerCase().includes("no information") ||
          response.toLowerCase().includes("no tengo")
        ) {
          console.log("⚠️  🔴 WARNING: Model indicates no information found");
          console.log("   💡 Suggestion: Check context relevance and similarity scores");
        } else {
          console.log("✅ 🟢 Model successfully used the provided context");
          
          // Check if response mentions sources
          const mentionsSources = response.toLowerCase().includes("readme") || 
                                response.toLowerCase().includes("source") ||
                                response.toLowerCase().includes("file");
          console.log(`📚 Source attribution: ${mentionsSources ? '✅' : '❌'}`);
        }
        
        console.log("═".repeat(80));
      } else {
        console.log("\n💡 Answer:", response);
      }
      
    } catch (error) {
      console.error("❌ Error processing question:", error.message);
    }
  }

  // 5. Interactive Chat Function
  async function startInteractiveChat() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });

    console.log("\n" + "═".repeat(80));
    console.log("🤖 RSK-CLI INTERACTIVE CHAT");
    console.log("═".repeat(80));
    console.log("💬 Ask me anything about rsk-cli!");
    console.log("📝 Type 'help' for example questions");
    console.log("🚪 Type 'exit' or 'quit' to end the session");
    console.log(`📚 FAQ System: ${faqStats.questions} instant answers available`);
    if (DEBUG_MODE) {
      console.log("🔍 DEBUG MODE: Full analytics enabled");
    }
    if (ULTRA_FAST) {
      console.log("🚀 ULTRA FAST MODE: Maximum speed optimizations (minimal context, ultra-short prompts)");
    } else if (FAST_MODE) {
      console.log("⚡ FAST MODE: Optimized for speed (smaller context, caching enabled)");
    }
    if (CACHE_EMBEDDINGS) {
      const modeText = ULTRA_FAST ? "Ultra Fast Mode" : (FAST_MODE ? "Fast Mode" : "Normal Mode");
      console.log(`🚀 PERFORMANCE: Embedding caching enabled (${modeText})`);
    }
    console.log("═".repeat(80));

    const askInteractiveQuestion = () => {
      rl.question("\n❓ Your question: ", async (question) => {
        const trimmedQuestion = question.trim();
        
        if (trimmedQuestion.toLowerCase() === 'exit' || trimmedQuestion.toLowerCase() === 'quit') {
          console.log("\n👋 Thanks for using rsk-cli chat! Goodbye!");
          rl.close();
          return;
        }
        
        if (trimmedQuestion.toLowerCase() === 'help') {
          console.log("\n💡 Example questions you can ask:");
          console.log("  • What is rsk-cli?");
          console.log("  • How do I check my wallet balance?");
          console.log("  • How can I transfer RBTC?");
          console.log("  • What commands are available for smart contracts?");
          console.log("  • How do I create a new wallet?");
          console.log("  • What's the difference between mainnet and testnet?");
          console.log("  • Show me the exact command structure for wallet management");
          console.log("  • What are the available options for the transfer command?");
          askInteractiveQuestion();
          return;
        }
        
        if (trimmedQuestion === '') {
          console.log("⚠️  Please enter a question or type 'help' for examples.");
          askInteractiveQuestion();
          return;
        }

        try {
          if (!DEBUG_MODE) {
            if (ULTRA_FAST) {
              console.log("\n🚀 Ultra fast processing mode...");
            } else if (FAST_MODE) {
              console.log("\n⚡ Fast processing mode...");
            } else {
              console.log("\n🔄 Processing your question...");
            }
          }
          await askQuestion(trimmedQuestion);
        } catch (error) {
          console.error("❌ Error processing your question:", error.message);
        }
        
        askInteractiveQuestion();
      });
    };

    askInteractiveQuestion();
  }

  // 6. Example usage - Multiple questions to verify comprehension
  if (DEBUG_MODE) {
    console.log("\n" + "=".repeat(60));
    console.log("🧪 TESTING: Verifying model comprehension");
    console.log("=".repeat(60));
    
    // First, a simple question to verify basic operation
    console.log("\n🔬 INITIAL CHECK: Verifying basic operation");
    console.log("─".repeat(40));
    await askQuestion("What is rsk-cli?");
    
    console.log("\n⏳ Waiting 3 seconds before running the full test...");
    await new Promise(resolve => setTimeout(resolve, 3000));
  }
  
  const testQuestions = [
    "What is rsk-cli and what is it used for?",
    "What is the command to check my wallet balance using rsk-cli?",
    "How can I transfer RBTC using rsk-cli?",
    "What command do I use to deploy a smart contract?",
    "How do I verify a smart contract on Rootstock?",
    "What is the difference between mainnet and testnet in rsk-cli?",
    "How can I create a new wallet with rsk-cli?",
    "What does the bridge command do in rsk-cli?",
    "Show me the exact command structure for wallet management",
    "What are the available options for the transfer command?",
    // New code-specific questions
    "What are the exact function parameters for the balance command implementation?",
    "Show me the specific validation logic used in the transfer command",
    "What error handling is implemented in the wallet creation process?",
    "How does the deploy command handle ABI and bytecode validation?",
    "What are the exact TypeScript types used for command options?",
    "Show me the specific network configuration logic in the source code",
    "What are the exact validation rules for wallet addresses in the code?",
    "How does the bridge command handle different network types?",
    "What are the specific gas estimation algorithms used?",
    "Show me the exact error messages defined in the source code"
  ];
  
  // Choose execution mode based on environment variables
  if (INTERACTIVE) {
    // Interactive chat mode (default)
    await startInteractiveChat();
  } else if (TEST_MODE) {
    // Run all test questions mode
    const modeTitle = DEBUG_MODE ? "🧪 COMPREHENSIVE TESTING SUITE (WITH DEBUG)" : "🧪 COMPREHENSIVE TESTING SUITE";
    
    console.log("\n" + "═".repeat(80));
    console.log(modeTitle);
    console.log("═".repeat(80));
    console.log(`📋 Running ${testQuestions.length} test questions to verify model comprehension`);
    if (DEBUG_MODE) {
      console.log("🔍 Full analytics and diagnostics enabled");
    }
    console.log("═".repeat(80));
    
    for (let i = 0; i < testQuestions.length; i++) {
      const progressBar = "█".repeat(Math.round((i / testQuestions.length) * 30)) + "░".repeat(30 - Math.round((i / testQuestions.length) * 30));
      
      console.log(`\n┌${"─".repeat(78)}┐`);
      console.log(`│ ❓ QUESTION ${String(i + 1).padStart(2)}/${testQuestions.length} ${"".padEnd(59)} │`);
      console.log(`│ Progress: [${progressBar}] ${Math.round((i / testQuestions.length) * 100)}% ${"".padEnd(15)} │`);
      console.log(`└${"─".repeat(78)}┘`);
      console.log(`📝 "${testQuestions[i]}"`);
      console.log("─".repeat(80));
      
      await askQuestion(testQuestions[i]);
      
      // Pause between questions for readability
      if (i < testQuestions.length - 1) {
        console.log("\n⏳ Waiting 2 seconds before the next question...");
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    console.log(`\n┌${"─".repeat(78)}┐`);
    console.log(`│ ✅ TESTING COMPLETE - All ${testQuestions.length} questions processed ${"".padEnd(28)} │`);
    console.log(`│ 🎯 Model comprehension verification finished ${"".padEnd(32)} │`);
    console.log(`└${"─".repeat(78)}┘`);
  }
  
} catch (error) {
  if (error.code === 'ENOENT') {
    console.error("❌ Error: ./docs/README.md not found");
    console.error("💡 Make sure the file exists at the correct path");
  } else {
    console.error("❌ Error while processing:", error.message);
    console.error("Details:", error);
  }
}
