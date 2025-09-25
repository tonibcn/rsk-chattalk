// FAQ System for instant command responses
import fs from 'fs';

class FAQSystem {
  constructor() {
    this.faq = null;
    this.loadFAQ();
  }

  loadFAQ() {
    try {
      const faqData = fs.readFileSync('./docs/faq-commands.json', 'utf8');
      this.faq = JSON.parse(faqData);
    } catch (error) {
      console.log('⚠️ Could not load FAQ data, falling back to RAG only');
      this.faq = { faq: {}, general: {}, networks: {} };
    }
  }

  // Calculate similarity between question and FAQ entries
  calculateSimilarity(question, faqQuestion) {
    const q1 = question.toLowerCase().trim();
    const q2 = faqQuestion.toLowerCase().trim();
    
    // Exact match (highest priority)
    if (q1 === q2) return 1.0;
    
    // Contains match
    if (q1.includes(q2) || q2.includes(q1)) return 0.9;
    
    // Word-based similarity
    const words1 = q1.split(/\s+/);
    const words2 = q2.split(/\s+/);
    
    const intersection = words1.filter(word => 
      words2.some(w2 => w2.includes(word) || word.includes(w2))
    );
    
    const similarity = intersection.length / Math.max(words1.length, words2.length);
    return similarity;
  }

  // Find best matching FAQ entry
  findBestMatch(question) {
    let bestMatch = null;
    let bestScore = 0;
    const threshold = 0.4; // Minimum similarity threshold

    // Search through all FAQ categories
    const allCategories = { ...this.faq.faq, ...this.faq };
    
    for (const [categoryKey, category] of Object.entries(allCategories)) {
      if (!category.questions) continue;
      
      for (const faqQuestion of category.questions) {
        const score = this.calculateSimilarity(question, faqQuestion);
        
        if (score > bestScore && score >= threshold) {
          bestScore = score;
          bestMatch = {
            category: categoryKey,
            answer: category.answer,
            command: category.command || null,
            executable: category.executable || false,
            score: score,
            matchedQuestion: faqQuestion
          };
        }
      }
    }

    return bestMatch;
  }

  // Get instant answer if available, otherwise return null for RAG fallback
  async getInstantAnswer(question) {
    const match = this.findBestMatch(question);
    
    if (!match) {
      return null; // No FAQ match, use RAG
    }

    // Return instant FAQ response
    return {
      type: 'faq',
      answer: match.answer,
      command: match.command,
      executable: match.executable,
      confidence: match.score,
      category: match.category,
      matchedQuestion: match.matchedQuestion
    };
  }

  // Get command suggestion for execution
  getCommandSuggestion(question) {
    const match = this.findBestMatch(question);
    
    if (match && match.executable && match.command) {
      return {
        command: match.command,
        category: match.category,
        confidence: match.score
      };
    }
    
    return null;
  }

  // Get statistics about FAQ coverage
  getStats() {
    let totalQuestions = 0;
    let totalCategories = 0;
    
    const allCategories = { ...this.faq.faq, ...this.faq };
    
    for (const [key, category] of Object.entries(allCategories)) {
      if (category.questions) {
        totalCategories++;
        totalQuestions += category.questions.length;
      }
    }
    
    return {
      categories: totalCategories,
      questions: totalQuestions,
      commands: Object.keys(this.faq.faq || {}).length
    };
  }
}

export default FAQSystem;
