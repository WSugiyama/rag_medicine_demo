# 🧠 RAG-Powered PubMed Assistant

A lightweight Retrieval-Augmented Generation (RAG) prototype for searching integrative medicine research over ~1,870 PubMed abstracts (Bilingual interface with a measured cross-lingual retrieval gap) — powered by Milvus, PubMed abstracts, and flexible support for both cloud APIs (OpenAI) and local models (Ollama/Gemma 2). Supports semantic search and natural language Q&A in **English** and **Indonesian**.

---

## 📌 Project Status & Scope

- **Status:** Experimental research prototype (v0.1 baseline). Built to explore hybrid semantic + NER retrieval, bilingual (English/Indonesian) medical literature search, and local-vs-cloud LLM inference trade-offs.
- **Not Production or Clinical Software:** This repository is an educational/engineering demo—it has **not** been validated for clinical decision support, patient care, or production deployment. Do not use its outputs as medical advice.
- **Known Boundaries:** Current retrieval quality is strongest on broad English integrative-medicine queries (`0.54` mean Precision@5 on a 10-query sample) and remains early-stage on Indonesian queries (`0.10` mean Precision@5) and specific compound/safety queries (see [Evaluation & Empirical Findings](#evaluation-of-trade-offs--empirical-findings)).

## 🚀 Features

- 🔍 Semantic search over biomedical abstracts
- 🤖 Multilingual RAG assistant (English & Indonesian)
- 🧬 Embedding + vector indexing via Milvus
- 💬 Flexible backend: Run via OpenAI cloud models or **100% locally & for free via Ollama (Gemma 2 + nomic-embed-text)**
- 📚 **NEW**: Clickable source citations with PubMed and DOI links
- 🎯 **NEW**: Citation ranking by usage frequency in generated answers
- 🔍 **NEW**: Hybrid retrieval combining semantic similarity with NER-based keyword matching
- 💡 Built for researchers, students, and exploratory literature review


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
python3 tests/test_runner.py --suite indonesian	# Test multilingual extraction
python3 tests/test_runner.py --suite citations     # Test citation remapping & patterns
python3 tests/test_runner.py --suite term_extraction          # Test term extraction coverage
```

Test results are automatically serialized into timestamped artifacts (e.g., ```test_results_<timestamp>.json```) for continuous integration tracking.


#### Evaluation of Trade-offs & Empirical Findings

An ablation study across $n=8$ stratified queries ($5$ total execution repetitions per weight setting, with $1$ warmup trial discarded and $4$ measured trials recorded; see [`benchmarks/weight_sweep_results.json`](benchmarks/weight_sweep_results.json) for full per-query `top_3_doc_ids` and pairwise `rank_invariance` traces) surfaced key architectural insights and limitations:

1. **Cross-Lingual Retrieval Disparity:** English queries achieved substantially higher keyword-hit precision than Indonesian queries (`0.0`–`1.0` across Indonesian queries depending on whether English/shared biomedical terms like `COVID-19` appeared in abstracts). This highlights a cross-lingual semantic alignment bottleneck when using general embedding models (`nomic-embed-text`) on English-skewed biomedical corpora.
2. **Term-Match Precision Masks Document & Rank Shifts ($W_{sem} \in [0.30, 0.70]$):** Mean term-match `Precision@3` remained flat at **0.875** across $W_{sem} \in [0.30, 0.70]$, which at a glance looks like rank invariance. However, logging `top_3_doc_ids`, `kendall_tau`, and estimated context tokens shows that the keyword component *does* actively reorder and swap retrieved abstracts on specific multi-concept queries—such as *anti-inflammatory herbs for arthritis* (`top_3_jaccard = 0.50` across $0.30 \leftrightarrow 0.50 \leftrightarrow 0.70$, `kendall_tau = -0.3333` between $0.30$ and $0.70$) and *probiotics + COVID-19 vaccines* (swapping in PMID `42985158` at $W_{sem}=0.70$, shifting context size from `1170` to `1500` estimated tokens)—even when every swapped abstract still satisfies the binary term-match check. Other queries (e.g., *anxiety* and Indonesian *vaccine effectiveness*) were completely rank-invariant (`top_3_jaccard = 1.0`, `kendall_tau = 1.0`) across the entire sweep.
3. **High-Semantic-Weight Drift ($W_{sem} \ge 0.85$):** Mean `Precision@3` dropped from **0.875** to **0.792** at $W_{sem} \in \{0.85, 0.95\}$, driven by *probiotics for digestive health* dropping from `1.00` to `0.333` (sharing only `1/3` top-3 PMIDs with $W_{sem} \le 0.70$, `top_3_jaccard = 0.20`). At the same time, high semantic weights also triggered rank and document shifts on *arthritis* (`top_3_jaccard = 0.0` vs. $W_{sem}=0.30$), *meditation + immune function* ($W_{sem}=0.95$), and *Indonesian COVID-19 symptoms* ($W_{sem}=0.95$) without changing their binary term-match `Precision@3`.

#### 📊 Hybrid Weight Sweep Precision@3 ($n=8$ Queries)
| Semantic Weight ($W_{sem}$) | Keyword Weight ($W_{kw}$) | Mean Precision@3 | Top-3 Ranking & Context Stability |
| :---: | :---: | :---: | :--- |
| **0.30** | 0.70 | **0.875** | Baseline hybrid setting; favors lexical term overlap on multi-term queries. |
| **0.50** | 0.50 | **0.875** | Identical top-3 sets on 7/8 queries vs. $0.30$; swaps 1 doc on *arthritis* (`jaccard = 0.50`, $\tau = 0.667$). |
| **0.70** | 0.30 | **0.875** | Identical top-3 sets on 6/8 queries vs. $0.30$; reorders/swaps docs on *arthritis* ($\tau = -0.333$) and *probiotics + COVID-19* (`jaccard = 0.50`). |
| **0.85** | 0.15 | **0.792** | *Probiotics for digestive health* drops (`1.00` → `0.333`, `jaccard = 0.20`); *arthritis* top-3 set turns over completely vs. $0.30$ (`jaccard = 0.0`). |
| **0.95** | 0.05 | **0.792** | Near-pure dense retrieval; introduces additional top-3 swaps on *meditation* and *Indonesian COVID-19 symptoms* (`jaccard = 0.50`). |


#### 📊 Split Latency & Top-K Ablation Results (Local Prototype, Split Timing)
| Top-$K$ | End-to-End Retrieval Median | Generation Median (Local LLM) | Avg Tokens | Observed Trend / Hypothesis |
| :---: | :---: | :---: | :---: | :--- |
| **$K = 3$** | **0.0281s** (~28 ms) | **1.237s** | 28.7 | **Hypothesis:** Compact context window yields high precision with minimal token generation. |
| **$K = 5$** | **0.0294s** (~29 ms) | **5.817s** | 104.0 | **Observation:** Balanced context window; stable extraction across queries. |
| **$K = 10$** | **0.0329s** (~33 ms) | **6.831s** | 202.7 | **Hypothesis:** Broader recall increases context size, amplifying generation latency and token output. |

**Note**: Local inference latencies exhibit run-to-run variance depending on hardware cache state and token generation length.

#### 📊 Hand-Judged Precision@5 Baseline ($n=12$ Queries, 60 Judgments)
To establish a reliable human-judged baseline beyond automated weak labels, top-5 retrieved documents were manually evaluated across **12 queries** (**10 English**, **2 Indonesian**; **27/60 relevant**, overall **Mean Precision@5 = 0.45**; English mean = **0.54**, Indonesian mean = **0.10**).

| # | Query | Lang | Relevant / 5 | Precision@5 |
| :---: | :--- | :---: | :---: | :---: |
| 1 | How effective is acupuncture for chronic pain management? | `en` | 5/5 | **1.00** |
| 2 | What are the benefits of probiotics for digestive health? | `en` | 3/5 | **0.60** |
| 3 | How do anti-inflammatory herbs help with arthritis? | `en` | 5/5 | **1.00** |
| 4 | What natural treatments are effective for anxiety? | `en` | 1/5 | **0.20** |
| 5 | How does meditation affect immune system function? | `en` | 2/5 | **0.40** |
| 6 | Can yoga help with depression symptoms? | `en` | 2/5 | **0.40** |
| 7 | What dietary supplements support heart health? | `en` | 3/5 | **0.60** |
| 8 | How effective is turmeric for inflammation? | `en` | 1/5 | **0.20** |
| 9 | What are the health benefits of omega-3 fatty acids? | `en` | 5/5 | **1.00** |
| 10 | Are green tea extracts safe for long-term liver health? | `en` | 0/5 | **0.00** |
| 11 | Apa saja manfaat terapi relaksasi bagi penderita kecemasan kronis? | `id` | 0/5 | **0.00** |
| 12 | Bagaimana efektivitas pengobatan herbal terhadap nyeri sendi? | `id` | 1/5 | **0.20** |

See [`tests/manual_precision_results.json`](tests/manual_precision_results.json) for full per-rank relevance judgments.

**Key Takeaways from the Precision@5 Audit:**
- **Higher top-5 relevance on broad English queries in our 12-query sample:** Broad, well-represented topics such as acupuncture for chronic pain (`1.00`), anti-inflammatory herbs for arthritis (`1.00`), and omega-3 fatty acids (`1.00`) consistently surfaced relevant top-5 results.
- **Indonesian cross-lingual drift:** Non-English queries averaged **0.10** Precision@5, frequently pulling off-target conditions, non-clinical surveys, or tangentially related interventions.
- **Specific extract & condition mismatches:** Queries targeting specific substances or safety profiles (such as green tea extracts for liver health, `0.00`, or turmeric for inflammation, `0.20`) often returned broader multi-herb formulas, general reviews, or mechanistic/animal studies instead.

#### ⚠️ Evaluation Harness Limitations & Next Steps
- **Weak Label Bottleneck:** Initial Recall@K / MRR harness runs (scoring Recall@5 ~0.07 and MRR ~0.11) were heavily penalized by strict keyword-derived ground-truth labels rather than actual retrieval failure.
- **Baseline Established, Next Focus:** With the 12-query hand-judged Precision@5 baseline recorded (**0.45** overall), priority improvements are cross-lingual query normalization/expansion for Indonesian queries, entity/compound matching for specific extracts, and reranking/filtering out non-clinical or preclinical noise.


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

