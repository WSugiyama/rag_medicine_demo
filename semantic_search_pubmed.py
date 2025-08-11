import re
import spacy
from pymilvus import connections, Collection
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
COLLECTION_NAME = "PubMed_Integrative_v2"

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("⚠️ spaCy model not found. Installing...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")

# Load collection
collection = Collection(name=COLLECTION_NAME)
collection.load()

# Helper: Extract key medical terms from query using NER
def extract_medical_terms(query):
    """Extract key medical terms using Named Entity Recognition for better scalability"""
    doc = nlp(query.lower())
    
    # Extract entities that are relevant for medical search
    relevant_entities = []
    
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "CONDITION", "CHEMICAL", "TREATMENT", "PROCEDURE"]:
            relevant_entities.append(ent.text.lower())
        elif ent.label_ == "ORG" and any(org_term in ent.text.lower() for org_term in ["hospital", "clinic", "medical", "health"]):
            relevant_entities.append(ent.text.lower())
    
    # Also extract common medical prefixes/suffixes that might not be caught by NER
    medical_patterns = [
        r'\b(probiotic|antibiotic|anti-inflammatory|anti-viral)\b',
        r'\b(covid|coronavirus|sars-cov)\b',
        r'\b(herbal|natural|traditional|alternative)\b',
        r'\b(inflammation|immune|immunity)\b',
        r'\b(treatment|therapy|intervention|study)\b'
    ]
    
    pattern_terms = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, query.lower())
        pattern_terms.extend(matches)
    
    # Combine NER entities with pattern matches, removing duplicates
    all_terms = relevant_entities + pattern_terms
    return list(set(all_terms))

# Helper: Filter results by relevance using hybrid approach
def filter_results_by_relevance(results, query, medical_terms):
    """Filter results using hybrid approach: semantic similarity + keyword relevance"""
    if not medical_terms:
        return results
    
    scored_results = []
    
    for r in results:
        entity = r.entity
        abstract = getattr(entity, "abstract", "").lower()
        title = getattr(entity, "title", "").lower()
        
        # Semantic similarity score (from Milvus)
        semantic_score = 1.0 - r.distance
        
        # Keyword relevance score
        keyword_score = 0
        for term in medical_terms:
            if term in abstract:
                keyword_score += 2
            if term in title:
                keyword_score += 1
        
        # Hybrid score: combine semantic and keyword relevance
        hybrid_score = 0.7 * semantic_score + 0.3 * (keyword_score / len(medical_terms))
        
        scored_results.append((r, hybrid_score))
    
    # Sort by hybrid score and return results
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [r for r, score in scored_results]

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
    
    # Extract medical terms for relevance filtering
    medical_terms = extract_medical_terms(query)
    print(f"🔍 Searching for: {', '.join(medical_terms) if medical_terms else 'general medical research'}")
    
    # Get more results initially for filtering
    initial_k = min(top_k * 3, 15)
    results = collection.search(
        data=[query_vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=initial_k,
        output_fields=["title", "abstract", "lang", "pmid", "doi"]
    )
    
    if not results or not results[0]:
        print("⚠️ No results found for your query.")
        return
    
    # Filter results by relevance
    relevant_results = filter_results_by_relevance(results[0], query, medical_terms)
    
    # Take top_k most relevant results
    final_results = relevant_results[:top_k]
    
    if not final_results:
        print("⚠️ No relevant results found after filtering.")
        return
    
    print(f"✅ Found {len(final_results)} relevant sources")
    print(f"\n🔍 Results for: '{query}'\n")
    
    for i, hit in enumerate(final_results):
        entity = hit.entity
        title = getattr(entity, 'title', '')
        abstract = getattr(entity, 'abstract', '')
        lang = getattr(entity, 'lang', '')
        pmid = getattr(entity, 'pmid', '')
        doi = getattr(entity, 'doi', '')
        
        print(f"{i+1}. 🧠 {title} [{lang}]\n{abstract[:500]}...\n(score: {hit.distance:.4f})")
        
        # Add clickable links
        if pmid:
            print(f"   PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        if doi:
            print(f"   DOI: https://doi.org/{doi}")
        print()

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your medical search query: ")
    search(user_query, top_k=3)

