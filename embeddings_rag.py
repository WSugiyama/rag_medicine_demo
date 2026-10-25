from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
import openai
from openai import OpenAI
import os
import json
from dotenv import load_dotenv

load_dotenv()

# Check if running locally or in cloud mode
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"

if USE_LOCAL_LLM:
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    EMBEDDING_MODEL = "nomic-embed-text"
    VECTOR_DIM = 768  # nomic-embed-text dimension size
    print("💻 Running embedding generation LOCALLY via Ollama (nomic-embed-text)")
else:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
    VECTOR_DIM = 1536  # OpenAI text-embedding-3-small dimension size
    print("☁️ Running embedding generation via OpenAI cloud API")

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

COLLECTION_NAME = "PubMed_Integrative_v2"

# Drop collection if it exists
if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

# Define schema with dynamic dimension based on local/cloud choice
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="abstract", dtype=DataType.VARCHAR, max_length=4096),
    FieldSchema(name="lang", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="pmid", dtype=DataType.VARCHAR, max_length=32),
    FieldSchema(name="doi", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIM)
]
schema = CollectionSchema(fields, description="PubMed Integrative Medicine Collection v2")
collection = Collection(name=COLLECTION_NAME, schema=schema)

BATCH_SIZE = 32
# Load data
with open("pubmed_integrative.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

# Embed abstracts using the explicit client instance
def embed_texts(texts):
    response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    return [r.embedding for r in response.data]

# Prepare for insertion
texts = [doc["abstract"] for doc in docs]
titles = [doc["title"] for doc in docs]
langs = [doc["lang"] for doc in docs]
pmids = [doc["pmid"] for doc in docs]
dois = [doc.get("doi", "") for doc in docs]  

for i in range(0, len(texts), BATCH_SIZE):
    MAX_LEN = 2000
    batch_texts_raw = texts[i:i + BATCH_SIZE]
    batch_titles = titles[i:i + BATCH_SIZE]
    batch_langs = langs[i:i + BATCH_SIZE]

    # Truncate abstracts to fit schema constraint
    MAX_LEN = 3500  # Safe for 4096 byte VARCHAR limit
    batch_texts = [txt[:MAX_LEN] for txt in batch_texts_raw]


    try:
        batch_embeddings = embed_texts(batch_texts)
        batch_pmids = pmids[i:i + BATCH_SIZE]
        batch_dois = dois[i:i + BATCH_SIZE]
        entities = [batch_titles, batch_texts, batch_langs, batch_pmids, batch_dois, batch_embeddings]

        collection.insert(entities)
        collection.flush()
        print(f"✅ Inserted batch {i//BATCH_SIZE + 1}: {len(batch_texts)} docs")
    except Exception as e:
        print(f"❌ Error in batch {i//BATCH_SIZE + 1}: {e}")

index_params = {
    "metric_type": "COSINE",
    "index_type": "HNSW", 
    "params": {
        "M": 16,            
        "efConstruction": 200 
    }
}

# ✅ Build index before loading
try:
    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )
    print("✅ Index created on 'embedding' field")
except Exception as e:
    print(f"⚠️ Index creation failed: {e}")

# ✅ Only load collection after the index exists
try:
    collection.load()
    print(f"📦 Collection '{COLLECTION_NAME}' loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load collection: {e}")

