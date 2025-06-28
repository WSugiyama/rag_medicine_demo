import os
from dotenv import load_dotenv
from pymilvus import connections, Collection
import openai
import langdetect

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuration
COLLECTION_NAME = "PubMed_Integrative"
EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")  # Default to gpt-4o

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name=COLLECTION_NAME)

# Helper: Embed query
def embed_query(query):
    response = openai.embeddings.create(
        input=[query],
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

# Main RAG Answer Function
def generate_answer_with_rag(query, top_k=3, lang_filter=None):
    vector = embed_query(query)
    collection.load()

    expr = f'lang == "{lang_filter}"' if lang_filter else ""

    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        expr=expr,
        output_fields=["abstract", "title", "lang"]
    )

    if not results or not results[0]:
        print("⚠️ No results found for your query.")
        return

    passages = [r.entity.get("abstract") or "" for r in results[0]]
    question_lang = langdetect.detect(query)

    # Build prompt
    prompt = (
        f"You are a multilingual, world-class integrative medicine research assistant.\n"
        f"Question: {query}\n\n"
        f"Use the following context to answer:\n\n"
        + "\n\n".join(passages)
        + f"\n\nRespond in the same language as the question (detected: {question_lang})."
    )

    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    print("\n💬 RAG Answer:\n")
    print(response.choices[0].message.content.strip())

# CLI Entry Point
if __name__ == "__main__":
    user_query = input("🧠 Ask a medical research question: ").strip()
    # lang_filter = input("🌐 Optional language code filter (e.g., en, id): ").strip() or None
    generate_answer_with_rag(user_query)

