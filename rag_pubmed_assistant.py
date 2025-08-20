import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
import os
import re
from dotenv import load_dotenv
from pymilvus import connections, Collection
import openai
import langdetect
import spacy

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load spaCy model for NER (fallback to English if specific model not available)
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

# Configuration
COLLECTION_NAME = "PubMed_Integrative_v2"
EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")  # Default to gpt-4o

# Standardized messages for no-result scenarios
NO_RESULTS_MESSAGE = "No results found for your query."
NO_RELEVANT_MESSAGE = "No relevant results found after filtering."

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

# Helper: Add citations to answer if missing
def add_citations_to_answer(answer, sources):
    """Add citations to key statements in the answer if none are present"""
    if not sources:
        return answer
    # Do not add citations to standardized no-result messages
    if answer.strip() in {NO_RESULTS_MESSAGE, NO_RELEVANT_MESSAGE}:
        return answer
    
    # Split answer into sentences
    sentences = answer.split('. ')
    cited_answer = []
    next_citation = 1
    total_citations = len(sources)

    for sentence in sentences:
        cleaned = sentence.strip()
        if not cleaned:
            continue
        if next_citation <= total_citations:
            cited_answer.append(f"{cleaned} [{next_citation}]")
            next_citation += 1
        else:
            cited_answer.append(cleaned)

    return '. '.join(cited_answer)

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
        r'\b(probiotic|probiotics|antibiotic|antibiotics|anti-inflammatory|anti-viral)\b',
        r'\b(covid|coronavirus|sars-cov)\b',
        r'\b(herbal|natural|traditional|alternative)\b',
        r'\b(inflammation|inflammatory|immune|immunity|cardiovascular|digestive)\b',
        r'\b(treatment|therapy|intervention|study|supplement|supplements)\b',
        r'\b(turmeric|acupuncture|meditation|yoga|arthritis|anxiety|depression)\b',
        r'\b(omega[-\s]?3|omega3|fatty acids)\b',
    ]
    
    pattern_terms = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, query.lower())
        pattern_terms.extend(matches)
    
    # Combine NER entities with pattern matches, removing duplicates
    all_terms = relevant_entities + pattern_terms
    return list(set(all_terms))

# Helper: Filter results by relevance
def filter_results_by_relevance(results, query, medical_terms):
    """Filter results using hybrid approach: semantic similarity + keyword relevance"""
    if not medical_terms:
        return results  # Return all if no medical terms extracted
    
    scored_results = []
    query_lower = query.lower()
    
    for r in results:
        entity = r.entity
        abstract = getattr(entity, "abstract", "").lower()
        title = getattr(entity, "title", "").lower()
        
        # Semantic similarity score (from Milvus)
        semantic_score = 1.0 - r.distance  # Convert distance to similarity
        
        # Keyword relevance score
        keyword_score = 0
        for term in medical_terms:
            if term in abstract:
                keyword_score += 2  # Abstract match is more important
            if term in title:
                keyword_score += 1  # Title match
        
        # Hybrid score: combine semantic and keyword relevance
        # Weight semantic similarity more heavily (70%) than keyword matching (30%)
        hybrid_score = 0.7 * semantic_score + 0.3 * (keyword_score / len(medical_terms))
        
        scored_results.append((r, hybrid_score))
    
    # Sort by hybrid score and return results
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [r for r, score in scored_results]

# Helper: Reorder sources by citation frequency in the answer
def reorder_sources_by_usage(answer, sources, final_results, source_metadata):
    """Reorders sources based on how many times they are cited in the answer.
    Returns (reordered_sources_lines, citation_mapping, reordered_metadata).
    """
    citation_counts = {}
    
    # Count citations in the answer using regex to avoid partial matches
    for i in range(len(sources)):
        citation_pattern = rf"\[{i+1}\](?!\d)"
        count = len(re.findall(citation_pattern, answer))
        citation_counts[i+1] = count
    
    # Sort source indices by usage count (desc)
    sorted_citations = sorted(citation_counts.items(), key=lambda x: x[1], reverse=True)

    reordered_sources = []
    reordered_metadata = []
    citation_mapping = {}  # old_num -> new_num

    # Build reordered outputs and mapping
    for new_num, (old_num, count) in enumerate(sorted_citations, 1):
        citation_mapping[old_num] = new_num
        # Find metadata with original index
        meta = next((m for m in source_metadata if m["index"] == old_num), None)
        if meta is None:
            continue
        reordered_metadata.append({**meta, "index": new_num})
        reordered_sources.append(format_source_line(meta, new_num))

    return reordered_sources, citation_mapping, reordered_metadata

# Helper: Safely remap citations in answer text using regex
def remap_citations_in_answer(answer, citation_mapping):
    """Safely remap citation numbers using a two-phase placeholder strategy to avoid collisions."""
    # Phase 1: convert all citations to collision-free placeholders using original numbers
    def to_placeholder(match: re.Match) -> str:
        original_num = match.group(1)
        return f"[__{original_num}__]"

    phase_one = re.sub(r"(?<!\d)\[(\d+)\](?!\d)", to_placeholder, answer)

    # Phase 2: replace placeholders with their mapped numbers (or original if unmapped)
    def from_placeholder(match: re.Match) -> str:
        original_num = int(match.group(1))
        new_num = citation_mapping.get(original_num, original_num)
        return f"[{new_num}]"

    updated_answer = re.sub(r"\[__(\d+)__\]", from_placeholder, phase_one)
    return updated_answer

# Helper: Format a source line from metadata
def format_source_line(meta, number):
    title = meta.get("title", "")
    lang = meta.get("lang", "")
    pmid = meta.get("pmid", "")
    doi = meta.get("doi", "")
    line = f"[{number}] {title} ({lang})"
    if pmid:
        line += f" — PubMed: https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    if doi:
        line += f" — DOI: https://doi.org/{doi}"
    return line

# Main RAG Answer Function
def generate_answer_with_rag(query, top_k=5, lang_filter=None):
    """Enhanced RAG with hybrid search and relevance filtering"""
    vector = embed_query(query)
    collection.load()
    
    # Extract medical terms for relevance filtering
    medical_terms = extract_medical_terms(query)
    print(f"🔍 Searching for: {', '.join(medical_terms) if medical_terms else 'general medical research'}")

    expr = f'lang == "{lang_filter}"' if lang_filter else ""

    # Get more results initially for filtering
    initial_k = min(top_k * 3, 15)
    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=initial_k,
        expr=expr,
        output_fields=["abstract", "title", "lang", "pmid", "doi"]
    )

    if not results or not results[0]:
        print(f"⚠️ {NO_RESULTS_MESSAGE}")
        return {"status": "no_results", "answer": NO_RESULTS_MESSAGE, "sources": []}

    # Filter results by relevance
    relevant_results = filter_results_by_relevance(results[0], query, medical_terms)
    
    # Take top_k most relevant results
    final_results = relevant_results[:top_k]
    
    if not final_results:
        print(f"⚠️ {NO_RELEVANT_MESSAGE}")
        return {"status": "no_results", "answer": NO_RELEVANT_MESSAGE, "sources": []}

    print(f"✅ Found {len(final_results)} relevant sources")
    
    # Build source metadata list and aligned passages before prompting
    question_lang = langdetect.detect(query)

    # Build source metadata list for citations (preserves final_results order)
    source_metadata = []
    for i, hit in enumerate(final_results):
        entity = hit.entity
        pmid = getattr(entity, "pmid", "")
        doi = getattr(entity, "doi", "")
        title = getattr(entity, "title", "")
        lang = getattr(entity, "lang", "")
        abstract = getattr(entity, "abstract", "")
        
        source_metadata.append({
            "index": i + 1,  # original number
            "pmid": pmid,
            "doi": doi,
            "title": title,
            "lang": lang,
            "abstract": abstract,
        })

    # Build sources lines for the initial prompt numbering
    sources = [format_source_line(meta, meta["index"]) for meta in source_metadata]

    # Passages aligned to the same order as sources
    passages = [m["abstract"] for m in source_metadata]

    # Build prompt
    prompt = (
        f"You are a multilingual, world-class integrative medicine research assistant.\n"
        f"Question: {query}\n\n"
        f"Use the following context to answer:\n\n"
        + "\n\n".join(passages)
        + f"\n\nIMPORTANT: You MUST cite sources using [1], [2], [3], etc. throughout your answer. "
        f"Every claim or fact you mention should be backed by a citation from the provided sources. "
        f"Respond in the same language as the question (detected: {question_lang}).\n\n"
        f"Available sources:\n" + "\n".join(sources)
    )

    response = openai.chat.completions.create(
        model=GPT_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()
    
    # Post-process to ensure citations are present
    if not any(f"[{i+1}]" in answer for i in range(len(sources))):
        # If no citations found, add them to key statements
        print("⚠️ No citations detected in response. Adding citations to key statements...")
        answer = add_citations_to_answer(answer, sources)

    print("\n💬 RAG Answer:\n")
    print(answer)

    # Print Sources section with clickable links (same order as passages/context)
    print("\n📚 Sources:\n")
    for line in sources:
        print(line)

    return {"status": "ok", "answer": answer, "sources": sources}

# CLI Entry Point
if __name__ == "__main__":
    user_query = input("🧠 Ask a medical research question: ").strip()
    # lang_filter = input("🌐 Optional language code filter (e.g., en, id): ").strip() or None
    generate_answer_with_rag(user_query)

