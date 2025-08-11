#!/usr/bin/env python3
"""
Rebuild PubMed Integrative Medicine Collection with new schema (v2)
Includes pmid and doi fields for citations
"""

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
import openai
import os
import json
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

COLLECTION_NAME = "PubMed_Integrative_v2"
EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")

# Drop collection if it exists
if utility.has_collection(COLLECTION_NAME):
    print(f"🗑️ Dropping existing collection: {COLLECTION_NAME}")
    utility.drop_collection(COLLECTION_NAME)

# Define schema with new fields
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536)
]
schema = CollectionSchema(fields, description="PubMed Integrative Medicine Collection v2")
collection = Collection(name=COLLECTION_NAME, schema=schema)

BATCH_SIZE = 32

# Load data
print("📚 Loading PubMed data...")
with open("pubmed_integrative.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"📊 Found {len(docs)} documents to process")

# Embed abstracts
def embed_texts(texts):
    response = openai.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [r.embedding for r in response.data]

# Prepare for insertion
texts = [doc["abstract"] for doc in docs]
titles = [doc["title"] for doc in docs]
langs = [doc["lang"] for doc in docs]
pmids = [doc["pmid"] for doc in docs]
dois = [doc.get("doi", "") for doc in docs]  # Handle cases where DOI might be None

print("🚀 Starting batch insertion...")

for i in range(0, len(texts), BATCH_SIZE):
    MAX_LEN = 2000
    batch_texts_raw = texts[i:i + BATCH_SIZE]
    batch_titles = titles[i:i + BATCH_SIZE]
    batch_langs = langs[i:i + BATCH_SIZE]
    batch_pmids = pmids[i:i + BATCH_SIZE]
    batch_dois = dois[i:i + BATCH_SIZE]

    # Truncate abstracts to fit schema constraint
    batch_texts = [txt[:MAX_LEN] for txt in batch_texts_raw]

    try:
        batch_embeddings = embed_texts(batch_texts)
        entities = [batch_titles, batch_texts, batch_langs, batch_pmids, batch_dois, batch_embeddings]

        collection.insert(entities)
        collection.flush()
        print(f"✅ Inserted batch {i//BATCH_SIZE + 1}: {len(batch_texts)} docs")
    except Exception as e:
        print(f"❌ Error in batch {i//BATCH_SIZE + 1}: {e}")

# Build index
print("🔍 Building index...")
index_params = {
    "metric_type": "COSINE",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

try:
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("✅ Index created on 'embedding' field")
except Exception as e:
    print(f"⚠️ Index creation failed: {e}")

# Load collection
try:
    collection.load()
    print(f"📦 Collection '{COLLECTION_NAME}' loaded successfully.")
    print(f"🎉 Ready to use with citations! Run 'python3 rag_pubmed_assistant.py' to test.")
except Exception as e:
    print(f"❌ Failed to load collection: {e}")
