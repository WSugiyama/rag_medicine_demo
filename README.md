# 🧠 RAG-Powered PubMed Assistant

A lightweight, multilingual Retrieval-Augmented Generation (RAG) prototype for searching integrative medicine research — powered by Milvus, PubMed abstracts, and flexible support for both cloud APIs (OpenAI) and local models (Ollama/Gemma 2). Supports semantic search and natural language Q&A in **English** and **Indonesian**.

---

## 🚀 Features

- 🔍 Semantic search over biomedical abstracts
- 🤖 Multilingual RAG assistant (English & Indonesian)
- 🧬 Embedding + vector indexing via Milvus
- 💬 Flexible backend: Run via OpenAI cloud models or **100% locally & for free via Ollama (Gemma 2 + nomic-embed-text)**
- 📚 **NEW**: Clickable source citations with PubMed and DOI links
- 🎯 **NEW**: Citation ranking by usage frequency in generated answers
- 🔍 **NEW**: Hybrid retrieval combining semantic similarity with NER-based keyword matching
- 💡 Designed for researchers, clinicians, and curious minds

---

## 🛠️ Setup

### 1. Clone the Repo
```bash
git clone [https://github.com/WSugiyama/rag_medicine_demo.git](https://github.com/WSugiyama/rag_medicine_demo.git)
cd rag_medicine_demo

```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Copy and customize .env.example:

```bash
cp .env.example .env
```


### 4. Choose Your Execution Mode
#### Option A: Cloud Mode (OpenAI + Milvus)
Ensure your .env contains your OpenAI API key and use the default settings.

#### Option B: Local Mode (Ollama + Gemma 2 + Milvus — Zero API Cost) 💻
To run this project completely offline:

1. **Install and Run Ollama:**
   Download [Ollama](https://ollama.com/) and pull the Gemma 2 model in your terminal:

   ```Bash
   ollama pull gemma2
   ollama pull nomic-embed-text
   ```

### 5. Build Collection & Generate Embeddings
Run ```embeddings_rag.py``` to ingest and embed your PubMed dataset. It automatically matches dimensions whether you are using OpenAI (1536-dim) or local Ollama (nomic-embed-text 768-dim):

```Bash
# If running locally:
USE_LOCAL_LLM=true python3 embeddings_rag.py

# If running via cloud:
python3 embeddings_rag.py
```

## 💬 Example Query
```Bash
# Run via cloud:
python3 rag_pubmed_assistant.py

# Or run locally:
USE_LOCAL_LLM=true python3 rag_pubmed_assistant.py

🧠 Ask a medical research question: What herbal treatments help with inflammation?

```

The assistant provides answers with inline citations ```[1], [2], [3]``` and a Sources section containing clickable PubMed and DOI links.


## 🔐 Notes
* Input abstracts are truncated if too long (Milvus VARCHAR limit = 4096 chars).
* All documents are embedded and inserted in batches for performance.

## 🧪 Engineering Rigor & Testing Strategy

To maintain high reliability without the latency and cost of heavy LLM-based evaluation frameworks (like RAGAS or LLM-as-a-judge), this project implements a **lightweight, deterministic unit-testing framework** (`unittest`):

- **Domain Gatekeeper Pattern:** Non-medical queries (in both English and Indonesian) are intercepted by the term extraction layer and safely return an empty list (`[]`), preventing out-of-domain noise from polluting Milvus vector searches.
- **Multilingual NER & Regex Hybrid:** Combines spaCy entity recognition with bilingual (English/Indonesian) regex patterns to ensure precise biomedical keyword extraction across languages.
- **Structural Unit Tests:** Validates citation remapping, sequential integrity, and edge-case resilience with a **100% pass rate** on local test suites, keeping CI/CD feedback loops under 2 seconds.


## 🚀 Technical Improvements & Roadmap

### Current Enhancements
- **Citation Ranking**: Sources are now ordered by actual citation frequency in answers, not search result order
- **Hybrid Retrieval**: Combines semantic similarity (70%) with NER-based keyword relevance (30%)
- **Named Entity Recognition**: Uses spaCy for scalable medical term extraction instead of hardcoded patterns
- **Local Fallback Support**: Seamlessly switch between OpenAI APIs and local Ollama execution

### Future Roadmap
- **Cross-encoder Reranking**: Implement BERT-based reranking for final result ordering
- **BM25 Integration**: Add traditional keyword search as additional retrieval method
- **Domain Adaptation**: Fine-tune NER models on biomedical text for better entity recognition
- **Multi-modal Search**: Support for searching across different document types and formats

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

