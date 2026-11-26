import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
import os
import re
from dotenv import load_dotenv
from pymilvus import connections, Collection
import openai
from openai import OpenAI
import langdetect
import spacy

# Load environment variables
load_dotenv()

# Ollama local configuration support
if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
    openai.base_url = "http://localhost:11434/v1"
    openai.api_key = "ollama"


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

# Initialize the OpenAI client for local Ollama use
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # Ollama doesn't require a real key
)

# Configuration
COLLECTION_NAME = "PubMed_Integrative_v2"

# Automatically switch embedding model based on local/cloud mode
if os.getenv("USE_LOCAL_LLM", "false").lower() == "true":
    EMBEDDING_MODEL = "nomic-embed-text"
    GPT_MODEL = "gemma2"
    openai.base_url = "http://localhost:11434/v1"
    openai.api_key = "ollama"
else:
    EMBEDDING_MODEL = os.getenv("OPENAI_ENGINE", "text-embedding-3-small")
    GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o")


# Standardized messages for no-result scenarios
NO_RESULTS_MESSAGE = "No results found for your query."
NO_RELEVANT_MESSAGE = "No relevant results found after filtering."

# Connect to Milvus
connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name=COLLECTION_NAME)

# Helper: Embed query
def embed_query(query):
    response = client.embeddings.create(
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
    """Extract key medical terms using NER and bilingual patterns. 
    Returns an empty list if no medical terms are found (domain gatekeeper)."""
    doc = nlp(query.lower())
    
    # 1. Extract standard NER entities
    relevant_entities = []
    for ent in doc.ents:
        if ent.label_ in ["DISEASE", "CONDITION", "CHEMICAL", "TREATMENT", "PROCEDURE"]:
            relevant_entities.append(ent.text.lower())
        elif ent.label_ == "ORG" and any(term in ent.text.lower() for term in ["hospital", "clinic", "medical", "health"]):
            relevant_entities.append(ent.text.lower())
    
    # 2. Bilingual medical patterns (English + Indonesian)
    medical_patterns = [
        # English patterns
        r'\b(probiotic|probiotics|antibiotic|antibiotics|anti-inflammatory|anti-viral)\b',
        r'\b(covid|coronavirus|sars-cov)\b',
        r'\b(herbal|natural|traditional|alternative)\b',
        r'\b(inflammation|inflammatory|immune|immunity|cardiovascular|digestive)\b',
        r'\b(treatment|therapy|intervention|study|supplement|supplements)\b',
        r'\b(turmeric|acupuncture|meditation|yoga|arthritis|anxiety|depression)\b',
        r'\b(omega[-\s]?3|omega3|fatty acids)\b',
        # Indonesian patterns
        r'\b(vaksin|vaccine|varian|variant|gejala|symptom|pengobatan|terapi)\b',
        r'\b(kardiovaskular|imun|antibodi|probiotik|suplemen|herbal)\b'
    ]
    
    pattern_terms = []
    for pattern in medical_patterns:
        matches = re.findall(pattern, query.lower())
        pattern_terms.extend(matches)
    
    # Combine and de-duplicate (No risky fallback to general non-medical words)
    return list(set(relevant_entities + pattern_terms))

def compute_dynamic_weights(query, medical_terms, initial_distances):
    words = query.split()
    word_count = len(words) if words else 1

    # 1. Entity Density
    d_ner = len(medical_terms) / word_count

    # 2. Semantic Ambiguity via score variance (if results exist)
    if initial_distances:
        import numpy as np
        score_variance = np.var(initial_distances)
    else:
        score_variance = 0.1

    # Heuristic adjustment rule (anchored to empirical 0.30 / 0.70 baseline)
    if d_ner > 0.3:
        # High entity density -> lean even stronger on lexical keyword matching
        semantic_weight = 0.3
        keyword_weight = 0.7
    elif score_variance < 0.01:
        # Flat vector space -> balanced hybrid rather than 0.85 semantic drift
        semantic_weight = 0.5
        keyword_weight = 0.5
    else:
        # Default empirical optimal fallback
        semantic_weight = 0.3
        keyword_weight = 0.7

    return semantic_weight, keyword_weight


# Helper: Filter results using true weighted linear blending (preserves weight sweep)
def filter_results_by_relevance(results, query, medical_terms, semantic_weight=0.3, keyword_weight=0.7):
    """Blends semantic and keyword scores proportionally using active weight parameters."""
    if not results:
        return results  
    
    if not medical_terms:
        # Fallback to pure semantic sort if no medical terms extracted
        sorted_res = sorted(results, key=lambda r: r.distance)
        return sorted_res

    scored_results = []
    semantic_scores = []
    keyword_scores = []
    
    # 1. Collect raw scores
    for r in results:
        entity = r.entity
        abstract = getattr(entity, "abstract", "").lower()
        title = getattr(entity, "title", "").lower()
        
        sem_score = 1.0 - r.distance
        semantic_scores.append(sem_score)
        
        kw_hits = sum(2 for t in medical_terms if t in abstract) + sum(1 for t in medical_terms if t in title)
        keyword_scores.append(kw_hits)
        
    # 2. Normalize both components to [0.0, 1.0] so weights actually govern influence
    sem_min, sem_max = min(semantic_scores), max(semantic_scores)
    kw_min, kw_max = min(keyword_scores), max(keyword_scores)
    
    sem_range = sem_max - sem_min if sem_max > sem_min else 1.0
    kw_range = kw_max - kw_min if kw_max > kw_min else 1.0
    
    for i, r in enumerate(results):
        norm_sem = (semantic_scores[i] - sem_min) / sem_range
        norm_kw = (keyword_scores[i] - kw_min) / kw_range
        
        # Apply active weights from the sweep experiment
        hybrid_score = (semantic_weight * norm_sem) + (keyword_weight * norm_kw)
        scored_results.append((r, hybrid_score))
    
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [r for r, score in scored_results]

"""
# Helper: Filter results by relevance using Weighted Reciprocal Rank Fusion (RRF)
def filter_results_by_relevance(results, query, medical_terms, semantic_weight=0.7, keyword_weight=0.3):
    if not medical_terms or not results:
        return results  
    
    # 1. Rank by Semantic Distance (Milvus distance: lower distance = better rank #1)
    semantic_sorted = sorted(results, key=lambda r: r.distance)
    sem_ranks = {id(r): rank + 1 for rank, r in enumerate(semantic_sorted)}
    
    # 2. Rank by Keyword Hits (Higher hits = better rank #1)
    def compute_kw_hits(r):
        entity = r.entity
        abstract = getattr(entity, "abstract", "").lower()
        title = getattr(entity, "title", "").lower()
        return sum(2 for t in medical_terms if t in abstract) + sum(1 for t in medical_terms if t in title)
    
    keyword_sorted = sorted(results, key=compute_kw_hits, reverse=True)
    kw_ranks = {id(r): rank + 1 for rank, r in enumerate(keyword_sorted)}
    
    # 3. Compute Weighted RRF score
    k_rrf = 60
    scored_results = []
    
    for r in results:
        r_id = id(r)
        r_sem_rank = sem_ranks.get(r_id, len(results))
        r_kw_rank = kw_ranks.get(r_id, len(results))
        
        # Weighted RRF combination formula
        rrf_score = (semantic_weight * (1.0 / (k_rrf + r_sem_rank))) + \
                    (keyword_weight * (1.0 / (k_rrf + r_kw_rank)))
                    
        scored_results.append((r, rrf_score))
    
    # Sort by RRF score descending
    scored_results.sort(key=lambda x: x[1], reverse=True)
    return [r for r, score in scored_results]
"""

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
    
    # 🛑 DOMAIN GATEKEEPER: Refuse non-medical/out-of-domain queries
    if not medical_terms:
        print(f"🛑 Out-of-domain query detected: '{query}'. Refusing request.")
        
        # Determine language for a localized refusal message if needed, or default cleanly
        try:
            q_lang = langdetect.detect(query)
        except Exception:
            q_lang = "en"
            
        if q_lang == "id":
            refusal_msg = (
                "Mohon maaf, saya hanya dapat menjawab pertanyaan terkait riset medis dan biomedis PubMed "
                "(seperti penyakit, pengobatan, gejala, atau studi klinis). Saya tidak dapat membantu pertanyaan umum."
            )
        else:
            refusal_msg = (
                "I'm sorry, but I can only answer biomedical and medical research questions "
                "related to PubMed abstracts (e.g., diseases, treatments, symptoms, or clinical studies). "
                "I cannot assist with general knowledge or non-medical queries."
            )
            
        print(f"\n💬 RAG Answer:\n\n{refusal_msg}\n")
        print("\n📚 Sources:\n\nNone (Out-of-domain query filtered by Domain Gatekeeper)")
        return {"status": "out_of_domain", "answer": refusal_msg, "sources": []}

    print(f"🔍 Searching for: {', '.join(medical_terms)}")


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

    # Extract distances for semantic ambiguity variance calculation
    initial_distances = [hit.distance for hit in results[0]]

    semantic_weight, keyword_weight = compute_dynamic_weights(query, medical_terms, initial_distances)
    
    # Filter results by relevance
    relevant_results = filter_results_by_relevance(results[0], query, medical_terms, semantic_weight, keyword_weight)

    
    # Take top_k most relevant results
    final_results = relevant_results[:top_k]
    
    if not final_results:
        print(f"⚠️ {NO_RELEVANT_MESSAGE}")
        return {"status": "no_results", "answer": NO_RELEVANT_MESSAGE, "sources": []}

    print(f"✅ Found {len(final_results)} relevant sources")
    

    # Build source metadata list and aligned passages before prompting
    try:
        question_lang = langdetect.detect(query)
    except Exception:
        question_lang = "en"
        
    # if langdetect misidentifies short English text as Spanish/other
    if question_lang not in ["en", "id"] and all(ord(c) < 128 for c in query):
        question_lang = "en"

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


    # Define system behavior and language rules clearly
    system_prompt = (
        f"You are a multilingual, world-class integrative medicine research assistant. "
        f"You MUST respond in the exact same language as the user's question "
        f"(Target language code: {question_lang}). Do not switch languages."
    )

    # User message contains the actual task, context, and data
    user_content = (
        f"Question: {query}\n\n"
        f"Use the following context to answer:\n\n"
        + "\n\n".join(passages)
        + f"\n\nIMPORTANT: You MUST cite sources using [1], [2], [3], etc. throughout your answer. "
        f"Every claim or fact you mention should be backed by a citation from the provided sources.\n\n"
        f"Available sources:\n" + "\n".join(sources)
    )

    # Call local Ollama chat endpoint with proper role separation
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
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

