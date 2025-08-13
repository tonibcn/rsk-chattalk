// rag.js
import fs from "fs";
import { Ollama } from "@langchain/ollama";
import { Document } from "langchain/document";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OllamaEmbeddings } from "@langchain/ollama";

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
      console.log("📁 Reading source files (src)...");
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
              console.log(`  ✅ ${file} (${content.length} chars)`);
            } else {
              console.log(`  ⚠️  ${file} (too short: ${content.length} chars)`);
            }
          } catch (err) {
            console.log(`  ❌ Error reading ${file}: ${err.message}`);
          }
        }
      });
      
      console.log(`📁 Total source files read: ${documents.length}`);
      
      // Show some sample content
      if (documents.length > 0) {
        console.log("\n📝 Sample of first file:");
        const sample = documents[0].pageContent.substring(0, 200);
        console.log(`"${sample}..."`);
      }
    } else {
      console.log("⚠️ 'src' folder not found. Only README.md will be used");
    }
  } catch (error) {
    console.log(`⚠️  Error reading source folder: ${error.message}`);
  }
  
  return documents;
}

// 2. Load multiple information sources
try {
  const allDocs = [];
  
  // Read README.md
  console.log("📖 Reading README.md...");
  const readmeContent = fs.readFileSync("./docs/README.md", "utf8");
  allDocs.push(new Document({ 
    pageContent: readmeContent,
    metadata: { source: "README.md", type: "documentation" }
  }));
  console.log("✅ README.md read successfully, size:", readmeContent.length, "characters");
  
  // Read source files
  const commandDocs = readCommandsFolder();
  allDocs.push(...commandDocs);
  
  console.log(`📚 Total documents loaded: ${allDocs.length}`);
  
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, // Increased from 500 to better handle source code
    chunkOverlap: 100, // Increased overlap for better context
  });

  const docs = await splitter.splitDocuments(allDocs);
  console.log("📄 Chunks created:", docs.length);

  // 3. Create embeddings (in-memory, no external server)
  console.log("🔄 Creating embeddings with 'nomic-embed-text'...");
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
  console.log("✅ Embeddings created:", index.length);

  // 4. Ask function
  async function askQuestion(question) {
    try {
      console.log("🔍 Computing query embedding...");
      const queryVec = await embeddings.embedQuery(question);

      console.log("🔎 Searching similar documents...");
      const scored = index.map(({ vector, text, metadata }) => ({
        score: cosineSimilarity(queryVec, vector),
        text,
        metadata
      }));
      scored.sort((a, b) => b.score - a.score);
      const topK = scored.slice(0, 5); // Increased from 3 to 5 for better context

      console.log("📊 Top matches:");
      topK.forEach((item, i) => {
        console.log(`  ${i + 1}. Score: ${item.score.toFixed(4)} | Source: ${item.metadata.source}`);
        console.log(`     Preview: "${item.text.substring(0, 100)}..."`);
      });

      const context = topK.map(r => r.text).join("\n\n---\n\n");
      
      // Verify that the context contains useful information
      console.log(`\n📋 Total context length: ${context.length} chars`);
      console.log("🔍 Checking context content...");
      
      // Keyword scan
      const keywords = question.toLowerCase().split(' ').filter(word => word.length > 3);
      const contextLower = context.toLowerCase();
      const foundKeywords = keywords.filter(keyword => contextLower.includes(keyword));
      
      console.log(`🎯 Keywords found in context: ${foundKeywords.join(', ')}`);
      
      // Special debugging for balance-related questions
      if (question.toLowerCase().includes('balance')) {
        console.log("\n🔍 Special DEBUG for balance question:");
        console.log("📁 Looking for balance-specific content in context...");
        
        const balanceContent = contextLower.includes('balance') ? '✅' : '❌';
        const commandContent = contextLower.includes('command') ? '✅' : '❌';
        const walletContent = contextLower.includes('wallet') ? '✅' : '❌';
        
        console.log(`  Balance: ${balanceContent}`);
        console.log(`  Command: ${commandContent}`);
        console.log(`  Wallet: ${walletContent}`);
        
        // Show specific balance-related snippets
        const balanceSnippets = context.split('\n').filter(line => 
          line.toLowerCase().includes('balance') || 
          line.toLowerCase().includes('command') ||
          line.toLowerCase().includes('wallet')
        );
        
        if (balanceSnippets.length > 0) {
          console.log("\n📝 Relevant snippets found:");
          balanceSnippets.slice(0, 3).forEach((snippet, i) => {
            console.log(`  ${i + 1}. "${snippet.trim()}"`);
          });
        }
      }
      
      if (foundKeywords.length === 0) {
        console.log("⚠️  WARNING: No relevant keywords found in context");
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

      console.log("🤖 Generating answer...");
      const response = await ollamaModel.call(prompt);
      console.log("🤖 Answer:", response);
      
      // Verificar si la respuesta menciona el contexto
      if (
        response.toLowerCase().includes("i don't have") ||
        response.toLowerCase().includes("no information") ||
        response.toLowerCase().includes("no tengo")
      ) {
        console.log("⚠️  WARNING: Model indicates no information. Check context.");
      } else {
        console.log("✅ Model appears to have used the provided context");
      }
      
    } catch (error) {
      console.error("❌ Error processing question:", error.message);
    }
  }

  // 5. Example usage - Multiple questions to verify comprehension
  console.log("\n" + "=".repeat(60));
  console.log("🧪 TESTING: Verifying model comprehension");
  console.log("=".repeat(60));
  
  // First, a simple question to verify basic operation
  console.log("\n🔬 INITIAL CHECK: Verifying basic operation");
  console.log("─".repeat(40));
  await askQuestion("What is rsk-cli?");
  
  console.log("\n⏳ Waiting 3 seconds before running the full test...");
  await new Promise(resolve => setTimeout(resolve, 3000));
  
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
  
  for (let i = 0; i < testQuestions.length; i++) {
    console.log(`\n${"─".repeat(40)}`);
    console.log(`❓ QUESTION ${i + 1}/${testQuestions.length}:`);
    console.log(`"${testQuestions[i]}"`);
    console.log(`${"─".repeat(40)}`);
    
    await askQuestion(testQuestions[i]);
    
    // Pause between questions for readability
    if (i < testQuestions.length - 1) {
      console.log("\n⏳ Waiting 2 seconds before the next question...");
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
  }
  
  console.log(`\n${"─".repeat(40)}`);
  console.log("✅ TESTING COMPLETE - Model comprehension check");
  console.log(`${"─".repeat(40)}`);
  
} catch (error) {
  if (error.code === 'ENOENT') {
    console.error("❌ Error: ./docs/README.md not found");
    console.error("💡 Make sure the file exists at the correct path");
  } else {
    console.error("❌ Error while processing:", error.message);
    console.error("Details:", error);
  }
}
