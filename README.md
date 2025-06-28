# 🧠 RAG-Powered PubMed Assistant

A lightweight, multilingual Retrieval-Augmented Generation (RAG) prototype for searching integrative medicine research — powered by OpenAI embeddings, Milvus, and PubMed abstracts. Supports semantic search and natural language Q&A in **English** and **Indonesian**.

---

## 🚀 Features

- 🔍 Semantic search over biomedical abstracts
- 🤖 Multilingual RAG assistant (English & Indonesian)
- 🧬 Embedding + vector indexing via Milvus
- 💬 Query understanding via OpenAI `text-embedding-3-small` and GPT models
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

## 💬 Example Query
```bash
python3 rag_pubmed_assistant.py
🧠 Ask a medical research question: What herbal treatments help with inflammation?
```

## 🔐 Notes
* Input abstracts are truncated if too long (Milvus VARCHAR limit = 2048 chars).
* All documents are embedded and inserted in batches for performance.


