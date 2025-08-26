// rag.js
import fs from "fs";
import { Ollama } from "@langchain/ollama";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";

// 0. Debug flag - set DEBUG_MODE=true node rag.js for detailed output
const DEBUG_MODE = process.env.DEBUG_MODE === "true";

// 1. Ollama configuration with timeouts
const ollamaModel = new Ollama({
  baseUrl: "http://localhost:11434",
  model: "llama3.2",
  timeout: 60000, // 60 segundos timeout
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
    chunkSize: 1000, // Increased from 500 to better handle source code
    chunkOverlap: 100, // Increased overlap for better context
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

  // 3. Create embeddings (in-memory, no external server)
  if (DEBUG_MODE) {
    console.log("\n🔄 Creating embeddings with 'nomic-embed-text'...");
    console.log("   ⏳ This may take a moment depending on document count...");
  }
  
  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text",
    baseUrl: "http://localhost:11434",
    timeout: 30000, // 30 seconds timeout
  });

  const docTexts = docs.map(d => d.pageContent);
  const docVectors = await embeddings.embedDocuments(docTexts);
  const index = docVectors.map((vec, i) => ({ 
    vector: vec, 
    text: docTexts[i],
    metadata: docs[i].metadata 
  }));
  
  if (DEBUG_MODE) {
    console.log("✅ Embeddings created successfully!");
    console.log("┌" + "─".repeat(50) + "┐");
    console.log("│ 🧠 EMBEDDING STATISTICS                        │");
    console.log("└" + "─".repeat(50) + "┘");
    console.log(`   📊 Total vectors: ${index.length}`);
    console.log(`   🔢 Vector dimensions: ${docVectors[0]?.length || 0}`);
    console.log(`   💾 Memory usage: ~${Math.round(index.length * (docVectors[0]?.length || 0) * 4 / 1024 / 1024)} MB`);
  }

  // 4. Ask function
  async function askQuestion(question) {
    try {
      if (DEBUG_MODE) console.log("\n❓ Question:", question);
      if (DEBUG_MODE) console.log("🔍 Computing query embedding...");
      const queryVec = await embeddings.embedQuery(question);

      if (DEBUG_MODE) console.log("🔎 Searching similar documents...");
      const scored = index.map(({ vector, text, metadata }) => ({
        score: cosineSimilarity(queryVec, vector),
        text,
        metadata
      }));
      scored.sort((a, b) => b.score - a.score);
      const topK = scored.slice(0, 5); // Increased from 3 to 5 for better context

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

      const prompt = `
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

      if (DEBUG_MODE) console.log("🤖 Generating answer...");
      const response = await ollamaModel.call(prompt);
      
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

  // 5. Example usage - Multiple questions to verify comprehension
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
  
  if (DEBUG_MODE) {
    // Run all test questions in debug mode
    console.log("\n" + "═".repeat(80));
    console.log("🧪 COMPREHENSIVE TESTING SUITE");
    console.log("═".repeat(80));
    console.log(`📋 Running ${testQuestions.length} test questions to verify model comprehension`);
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
  } else {
    // In normal mode, just run the first two questions as examples
    console.log("\n" + "═".repeat(60));
    console.log("📝 SAMPLE QUESTIONS");
    console.log("═".repeat(60));
    await askQuestion("What is rsk-cli and what is it used for?");
    await askQuestion("What is the command to check my wallet balance using rsk-cli?");
    console.log("\n" + "═".repeat(60));
    console.log("✅ Sample complete. Use DEBUG_MODE=true for full testing.");
    console.log("═".repeat(60));
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
