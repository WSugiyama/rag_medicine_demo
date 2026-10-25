import os
import re
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymilvus import connections, Collection
import openai
from openai import OpenAI

# Environment setup
load_dotenv()

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


# Lazy import of helpers to avoid heavy dependencies at import-time
def _try_import_rag_helpers():
    try:
        from rag_pubmed_assistant import (
            extract_medical_terms as _extract_medical_terms,
            filter_results_by_relevance as _filter_results_by_relevance,
            format_source_line as _format_source_line,
            add_citations_to_answer as _add_citations_to_answer,
        )
        return (
            _extract_medical_terms,
            _filter_results_by_relevance,
            _format_source_line,
            _add_citations_to_answer,
        )
    except Exception:
        # Minimal fallbacks
        def _fallback_extract_medical_terms(text: str) -> List[str]:
            patterns = [
                r"\b(probiotic|antibiotic|anti-inflammatory|anti-viral)\b",
                r"\b(covid|coronavirus|sars-cov)\b",
                r"\b(herbal|natural|traditional|alternative)\b",
                r"\b(inflammation|immune|immunity)\b",
                r"\b(treatment|therapy|intervention|study)\b",
            ]
            terms: List[str] = []
            for p in patterns:
                terms.extend(re.findall(p, text.lower()))
            return list(set(terms))

        def _fallback_filter_results_by_relevance(results, query, medical_terms):
            if not medical_terms:
                return results
            scored = []
            for r in results:
                entity = r.entity
                abstract = getattr(entity, "abstract", "").lower()
                title = getattr(entity, "title", "").lower()
                keyword_score = 0
                for term in medical_terms:
                    if term in abstract:
                        keyword_score += 2
                    if term in title:
                        keyword_score += 1
                semantic_score = 1.0 - r.distance
                hybrid = 0.7 * semantic_score + 0.3 * (keyword_score / max(1, len(medical_terms)))
                scored.append((r, hybrid))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [r for r, _ in scored]

        def _fallback_format_source_line(meta: Dict[str, str], number: int) -> str:
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

        def _fallback_add_citations_to_answer(answer: str, sources: List[str]) -> str:
            if not sources:
                return answer
            sentences = [s for s in answer.split('. ') if s.strip()]
            cited: List[str] = []
            for i, s in enumerate(sentences):
                if i < len(sources):
                    cited.append(f"{s} [{i+1}]")
                else:
                    cited.append(s)
            return '. '.join(cited)

        return (
            _fallback_extract_medical_terms,
            _fallback_filter_results_by_relevance,
            _fallback_format_source_line,
            _fallback_add_citations_to_answer,
        )

(
    extract_medical_terms,
    filter_results_by_relevance,
    format_source_line,
    add_citations_to_answer,
) = _try_import_rag_helpers()

_collection_cache: Optional[Collection] = None

def get_collection() -> Collection:
    global _collection_cache
    if _collection_cache is not None:
        return _collection_cache
    connections.connect(alias="default", host="localhost", port="19530")
    _collection_cache = Collection(name=COLLECTION_NAME)
    return _collection_cache


def embed_text(text: str) -> List[float]:
    response = openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def extract_paper_reference(query: str) -> Dict[str, str]:
    """Extract a PubMed paper reference (PMID, DOI, or title) from the user query.

    Returns a dict with keys: {"type": "pmid"|"doi"|"title"|"none", "value": str}
    """
    # PMID patterns
    pmid_match = re.search(r"(?:PMID|PubMed ID)[:\s]*([0-9]{6,10})", query, flags=re.IGNORECASE)
    if pmid_match:
        return {"type": "pmid", "value": pmid_match.group(1)}

    # DOI patterns (supports doi.org links and raw DOI)
    doi_link_match = re.search(r"doi\.org/(10\.\d{4,9}/\S+)", query, flags=re.IGNORECASE)
    if doi_link_match:
        return {"type": "doi", "value": doi_link_match.group(1)}
    doi_match = re.search(r"(?:DOI)[:\s]*(10\.\d{4,9}/\S+)", query, flags=re.IGNORECASE)
    if doi_match:
        return {"type": "doi", "value": doi_match.group(1)}

    # Title in quotes (prefer double quotes)
    title_match = re.search(r'"([^"]{5,200})"', query)
    if title_match:
        return {"type": "title", "value": title_match.group(1).strip()}

    return {"type": "none", "value": ""}


def get_paper_by_reference(reference: Dict[str, str]) -> Optional[Dict[str, str]]:
    """Retrieve paper metadata by PMID/DOI or estimate from title via nearest neighbor.

    Returns a dict with keys: title, abstract, lang, pmid, doi
    """
    collection = get_collection()
    collection.load()

    ref_type = reference.get("type")
    ref_value = reference.get("value", "")

    try:
        if ref_type in ("pmid", "doi") and ref_value:
            field = "pmid" if ref_type == "pmid" else "doi"
            # Use Milvus scalar query when possible
            items = collection.query(
                expr=f'{field} == "{ref_value}"',
                output_fields=["title", "abstract", "lang", "pmid", "doi"],
                limit=1,
            )
            if items:
                item = items[0]
                return {
                    "title": item.get("title", ""),
                    "abstract": item.get("abstract", ""),
                    "lang": item.get("lang", ""),
                    "pmid": item.get("pmid", ""),
                    "doi": item.get("doi", ""),
                }

        # Fallback: if we have a title candidate, do a nearest-neighbor search by text
        title_text = ref_value if ref_type == "title" else ref_value or ""
        if title_text:
            vector = embed_text(title_text)
            results = collection.search(
                data=[vector],
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=1,
                output_fields=["title", "abstract", "lang", "pmid", "doi"],
            )
            if results and results[0]:
                hit = results[0][0]
                entity = hit.entity
                return {
                    "title": getattr(entity, "title", ""),
                    "abstract": getattr(entity, "abstract", ""),
                    "lang": getattr(entity, "lang", ""),
                    "pmid": getattr(entity, "pmid", ""),
                    "doi": getattr(entity, "doi", ""),
                }
    except Exception:
        # Gracefully degrade to None; caller decides how to proceed
        return None

    return None


def find_similar_papers(reference_paper: Dict[str, str], top_k: int = 5, lang_filter: Optional[str] = None):
    """Find semantically similar papers to the given reference paper, with hybrid relevance filtering."""
    if not reference_paper:
        return []

    collection = get_collection()
    collection.load()

    pmid_to_exclude = reference_paper.get("pmid", "")
    title = reference_paper.get("title", "")
    abstract = reference_paper.get("abstract", "")

    combined_text = f"{title}. {abstract}".strip()
    if not combined_text:
        combined_text = title or abstract

    vector = embed_text(combined_text)

    expr = f'lang == "{lang_filter}"' if lang_filter else ""
    initial_k = min(top_k * 3, 15)
    results = collection.search(
        data=[vector],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=initial_k,
        expr=expr,
        output_fields=["title", "abstract", "lang", "pmid", "doi"],
    )

    if not results or not results[0]:
        return []

    # Exclude the reference paper by pmid if available
    filtered_hits = [hit for hit in results[0] if getattr(hit.entity, "pmid", "") != pmid_to_exclude]

    # Hybrid relevance filtering using extracted medical terms from the reference title/abstract
    medical_terms = extract_medical_terms(combined_text)
    relevant_results = filter_results_by_relevance(filtered_hits, combined_text, medical_terms)

    return relevant_results[:top_k]


def generate_similarity_analysis(query: str, reference_paper: Dict[str, str], similar_papers) -> Tuple[str, List[str]]:
    """Generate a comparative analysis answer between a reference paper and similar papers.

    Returns tuple: (answer_text, source_lines)
    """
    # Build metadata list: reference paper first, then similar papers
    source_metadata = []
    sources = []
    passages = []

    if reference_paper:
        source_metadata.append(
            {
                "index": 1,
                "pmid": reference_paper.get("pmid", ""),
                "doi": reference_paper.get("doi", ""),
                "title": reference_paper.get("title", ""),
                "lang": reference_paper.get("lang", ""),
                "abstract": reference_paper.get("abstract", ""),
            }
        )

    for i, hit in enumerate(similar_papers, start=2):
        entity = hit.entity
        meta = {
            "index": i,
            "pmid": getattr(entity, "pmid", ""),
            "doi": getattr(entity, "doi", ""),
            "title": getattr(entity, "title", ""),
            "lang": getattr(entity, "lang", ""),
            "abstract": getattr(entity, "abstract", ""),
        }
        source_metadata.append(meta)

    sources = [format_source_line(meta, meta["index"]) for meta in source_metadata]
    passages = [m.get("abstract", "") for m in source_metadata]

    prompt = (
        "You are a multilingual integrative medicine research assistant.\n"
        "The user references a specific PubMed paper and seeks similar/related literature.\n\n"
        f"User query: {query}\n\n"
        "Using the following abstracts (first is the reference paper), provide a concise comparative analysis:\n"
        "- Highlight main findings of the reference paper\n"
        "- Compare with similar papers on methodology, outcomes, and limitations\n"
        "- Identify consensus and disagreements\n"
        "- Suggest potential follow-up readings\n\n"
        + "\n\n".join(passages)
        + "\n\nIMPORTANT: You MUST cite sources as [1], [2], [3], etc. throughout your answer."
    )


# Point to local Ollama instance instead of OpenAI cloud
    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama" # Ollama doesn't require a real key
    )

    response = client.chat.completions.create(
        model="gemma2",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content.strip()

    # Ensure citations are present at minimum
    if not re.search(r"\[\d+\]", answer):
        answer = add_citations_to_answer(answer, sources)

    return answer, sources


def hybrid_paper_search(query: str, top_k: int = 5, lang_filter: Optional[str] = None) -> Optional[Dict[str, object]]:
    """Main entrypoint: parse reference, retrieve paper, find similars, and generate analysis."""
    reference = extract_paper_reference(query)

    # Try to get reference paper; if none, fall back to general semantic search using the full query
    reference_paper = get_paper_by_reference(reference)
    if not reference_paper:
        # Fallback: try to treat entire query as a title-like input
        vector = embed_text(query)
        collection = get_collection()
        results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=1,
            output_fields=["title", "abstract", "lang", "pmid", "doi"],
        )
        if results and results[0]:
            hit = results[0][0]
            entity = hit.entity
            reference_paper = {
                "title": getattr(entity, "title", ""),
                "abstract": getattr(entity, "abstract", ""),
                "lang": getattr(entity, "lang", ""),
                "pmid": getattr(entity, "pmid", ""),
                "doi": getattr(entity, "doi", ""),
            }
        else:
            try:
                from rag_pubmed_assistant import NO_RESULTS_MESSAGE
            except Exception:
                NO_RESULTS_MESSAGE = "No results found for your query."
            return {"status": "no_results", "answer": NO_RESULTS_MESSAGE, "sources": []}

    similar_hits = find_similar_papers(reference_paper, top_k=top_k, lang_filter=lang_filter)
    if not similar_hits:
        try:
            from rag_pubmed_assistant import NO_RELEVANT_MESSAGE
        except Exception:
            NO_RELEVANT_MESSAGE = "No relevant results found after filtering."
        return {"status": "no_results", "answer": NO_RELEVANT_MESSAGE, "sources": []}
    answer, sources = generate_similarity_analysis(query, reference_paper, similar_hits)

    return {
        "reference": reference_paper,
        "similar_count": len(similar_hits),
        "answer": answer,
        "sources": sources,
        "status": "ok",
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid PubMed paper similarity search")
    parser.add_argument("--query", required=True, help="User query that may reference a PubMed paper")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lang", default=None, help="Optional language filter")
    args = parser.parse_args()

    result = hybrid_paper_search(args.query, top_k=args.top_k, lang_filter=args.lang)
    if not result:
        print("⚠️ Could not identify a reference paper from the query.")
    else:
        print("\n💬 Analysis:\n")
        print(result["answer"])
        print("\n📚 Sources:")
        for line in result["sources"]:
            print(line)


