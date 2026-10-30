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

### 🧪 Engineering Rigor & Testing Strategy

To maintain high reliability without the latency and cost of heavy LLM-based evaluation frameworks (like RAGAS or LLM-as-a-judge), this project implements a modular, deterministic unit-testing framework (`unittest`):

- **Domain Gatekeeper Pattern:** Non-medical queries (in both English and Indonesian) are intercepted by the term extraction layer and safely return an empty list (`[]`), preventing out-of-domain noise from polluting Milvus vector searches.
- **Multilingual NER & Regex Hybrid:** Combines spaCy entity recognition with bilingual (English/Indonesian) regex patterns to ensure precise biomedical keyword extraction across languages.
- **Structural Unit Tests:** Validates citation remapping, sequential integrity, and edge-case resilience with a **100% pass rate** on local test suites, keeping CI/CD feedback loops under 2 seconds.

#### Running Unit Tests
You can execute the full test suite or target specific modules using command-line arguments:

```bash
# Run all test suites
python3 test_runner.py --suite all --verbose

# Run specific functional suites
python3 test_runner.py --suite indonesian   # Test multilingual extraction
python3 test_runner.py --suite citations    # Test citation remapping & patterns
python3 test_runner.py --suite perf         # Run performance benchmarks
```

Test results are automatically serialized into timestamped artifacts (e.g., ```test_results_<timestamp>.json```) for continuous integration tracking.

#### Evaluation of Trade-offs
Empirical testing reveals that retrieval performance and generation stability are highly dependent on query-type and local cache states. Rather than relying on a static $K$ value, robust production systems require dynamic top-$K$ selection or reranking layers.  See [```benchmark_results.json```](benchmark_results.json).

#### 📊 Empirical Top-$K$ Ablation Results (Local Gemma 2 / Ollama)
| Top-$K$ | Avg Latency | Behavioral Outcome / Observation |
| :---: | :---: | :--- |
| **$K = 3$** | ~9.70s | High precision; graceful handling of narrow contexts. |
| **$K = 5$** | ~5.51s | Fast token generation via rapid refusal short-circuiting on out-of-scope context matches. |
| **$K = 10$** | ~9.96s | Broad recall, but susceptible to **context drift** (introducing noisy peripheral papers). |

### 🚀 Technical Improvements & Roadmap

### Current Enhancements
- **Citation Ranking**: Sources are ordered by actual citation frequency in answers, not search result order.
- **Hybrid Retrieval**: Combines semantic similarity (70%) with NER-based keyword relevance (30%).
- **Named Entity Recognition**: Uses spaCy for scalable medical term extraction instead of hardcoded patterns.
- **Local Fallback Support**: Seamlessly switch between OpenAI APIs and local Ollama execution.

### Future Roadmap & Architecture Vision
- **Cross-Encoder Reranking (e.g., `bge-reranker` or `ms-marco-MiniLM`)**: 
  * *Why:* While dense vector search efficiently retrieves top-$K$ candidates, larger contexts ($K=10$) often introduce semantic noise and cause context drift. Cross-encoders will evaluate deep query-document interaction scores simultaneously *before* hitting the local LLM.
- **BM25 Lexical Hybrid Integration**: 
  * *Why:* Bridge the biomedical vocabulary gap for precise keyword matching (e.g., specific drug identifiers, gene symbols, or clinical abbreviations) that pure dense embedding models occasionally miss.
- **Domain Adaptation**: Fine-tune NER models on specialized biomedical corpora for higher entity extraction accuracy.
- **Multi-Modal Search**: Support for searching across mixed document schemas and clinical tables.


---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).

