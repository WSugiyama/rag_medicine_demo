from pymilvus import connections, Collection
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
COLLECTION_NAME = "PubMed_Integrative"

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Load collection
collection = Collection(name=COLLECTION_NAME)
collection.load()

# Embed the query
def embed_query(query):
    response = openai.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Search function
def search(query, top_k=3):
    query_vector = embed_query(query)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["title", "abstract", "lang"]
    )
    print(f"\n🔍 Results for: '{query}'\n")
    for i, hit in enumerate(results[0]):
        entity = hit.entity
        print(f"{i+1}. 🧠 {entity.get('title')} [{entity.get('lang')}]\n{entity.get('abstract')[:500]}...\n(score: {hit.distance:.4f})\n")

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your medical search query: ")
    search(user_query, top_k=3)

