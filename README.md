# 🧠 RAG-Powered PubMed Assistant

A lightweight, multilingual Retrieval-Augmented Generation (RAG) prototype for searching integrative medicine research — powered by OpenAI embeddings, Milvus, and PubMed abstracts. Supports semantic search and natural language Q&A in **English** and **Indonesian**.

---

## 🚀 Features

- 🔍 Semantic search over biomedical abstracts
- 🤖 Multilingual RAG assistant (English & Indonesian)
- 🧬 Embedding + vector indexing via Milvus
- 💬 Query understanding via OpenAI `text-embedding-3-small` and GPT models
- 📚 **NEW**: Clickable source citations with PubMed and DOI links
- 🎯 **NEW**: Citation ranking by usage frequency in generated answers
- 🔍 **NEW**: Hybrid retrieval combining semantic similarity with NER-based keyword matching
- 💡 Designed for researchers, clinicians, and curious minds

---

## 🛠️ Setup

### 1. Clone the Repo
```bash
git clone https://github.com/WSugiyama/rag_medicine_demo.git
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

You'll need:
* OpenAI API Key
* Milvus running locally (or via Docker)

### 4. Rebuild Collection with Citations (v2)
The new version includes clickable source citations. Rebuild your collection:

```bash
python3 rebuild_collection.py
```

This creates a new `PubMed_Integrative_v2` collection with PMID and DOI fields for proper citations.

## 💬 Example Query
```bash
python3 rag_pubmed_assistant.py
🧠 Ask a medical research question: What herbal treatments help with inflammation?
```

The assistant now provides answers with inline citations [1], [2], [3] and a Sources section containing clickable PubMed and DOI links.

## 🔐 Notes
* Input abstracts are truncated if too long (Milvus VARCHAR limit = 2048 chars).
* All documents are embedded and inserted in batches for performance.

## 🚀 Technical Improvements & Roadmap

### Current Enhancements
- **Citation Ranking**: Sources are now ordered by actual citation frequency in answers, not search result order
- **Hybrid Retrieval**: Combines semantic similarity (70%) with NER-based keyword relevance (30%)
- **Named Entity Recognition**: Uses spaCy for scalable medical term extraction instead of hardcoded patterns

### Future Roadmap
- **Cross-encoder Reranking**: Implement BERT-based reranking for final result ordering
- **BM25 Integration**: Add traditional keyword search as additional retrieval method
- **Domain Adaptation**: Fine-tune NER models on biomedical text for better entity recognition
- **Multi-modal Search**: Support for searching across different document types and formats


