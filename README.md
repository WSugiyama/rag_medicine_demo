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
- **Structural Unit Tests:** Validates citation remapping, sequential integrity, and edge-case resilience across **6 core test suites** with a **100% pass rate** (~1.8s local feedback loop including spaCy pipeline initialization).


#### Running Unit Tests
You can execute the full test suite or target specific modules using command-line arguments from the root directory:

```bash
# Run all test suites
time python3 tests/test_runner.py --suite all

# Run specific functional suites
python3 tests/test_runner.py --suite indonesian    # Test multilingual extraction
python3 tests/test_runner.py --suite citations     # Test citation remapping & patterns
python3 tests/test_runner.py --suite term_extraction          # Test term extraction coverage
```

Test results are automatically serialized into timestamped artifacts (e.g., ```test_results_<timestamp>.json```) for continuous integration tracking.


#### Evaluation of Trade-offs & Empirical Findings

An ablation study across $n=8$ stratified queries ($repeats=5$) surfaced key architectural insights and limitations:

1. **Cross-Lingual Retrieval Disparity:** English queries consistently achieved high precision (Precision@3 $\approx 1.0$), whereas Indonesian queries underperformed severely ($0.67$ and $0.0$). This highlights a cross-lingual semantic alignment bottleneck when using general embedding models (`nomic-embed-text`) on English-skewed biomedical corpora.
2. **Rank Invariance ($W_{sem} \in [0.3, 0.7]$):** Precision@3 and the resulting top-$3$ context sets remained entirely identical across this range. This indicates that the keyword component is currently inert in altering rank order over standard blending weights due to vector score magnitude dominance.
3. **Extreme Boundary Drop ($W_{sem} = 0.95$):** A precision drop was observed at the extreme upper bound, though this behavior is currently isolated to a single query instance ($n=1$) and requires broader validation before drawing general conclusions.

See [`benchmarks/weight_sweep_results.json`](benchmarks/weight_sweep_results.json) for full granular execution traces.

#### 📊 Hybrid Weight Sweep Precision@3 ($n=8$ Queries)
| Semantic Weight ($W_{sem}$) | Keyword Weight ($W_{kw}$) | Mean Precision@3 | Observation |
| :---: | :---: | :---: | :--- |
| **0.30** | 0.70 | ~0.79 | Heavy keyword weight; identical top-3 ranking output. |
| **0.50** | 0.50 | ~0.79 | Balanced hybrid weighting (rank-invariant to $0.3$). |
| **0.70** | 0.30 | ~0.79 | Standard experimental configuration. |
| **0.85** | 0.15 | ~0.79 | Near-pure vector search plateau. |
| **0.95** | 0.05 | ~0.71 | Single-query drop driven by loss of lexical constraint ($n=1$). |


#### 📊 Split Latency & Top-$K$ Ablation Results (Local Prototype, Split Timing)
| Top-$K$ | End-to-End Retrieval Median | Generation Median (Local LLM) | Avg Tokens | Observed Trend / Hypothesis |
| :---: | :---: | :---: | :---: | :--- |
| **$K = 3$** | **0.0281s** (~28 ms) | **1.237s** | 28.7 | **Hypothesis:** Compact context window yields high precision with minimal token generation. |
| **$K = 5$** | **0.0294s** (~29 ms) | **5.817s** | 104.0 | **Observation:** Balanced context window; stable extraction across queries. |
| **$K = 10$** | **0.0329s** (~33 ms) | **6.831s** | 202.7 | **Hypothesis:** Broader recall increases context size, amplifying generation latency and token output. |

**Note**: Local inference latencies exhibit run-to-run variance depending on hardware cache state and token generation length.


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

