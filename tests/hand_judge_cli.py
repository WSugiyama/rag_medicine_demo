import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance
from tests.eval_utils import get_git_commit, get_doc_id, precision_at_k

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")
collection.load()

BASE_DIR = os.path.dirname(__file__)
CACHE_PATH = os.path.join(BASE_DIR, "judgment_cache.json")
LEGACY_RESULTS_PATH = os.path.join(BASE_DIR, "manual_precision_results.json")
DATASET_PATH = os.path.join(BASE_DIR, "gold_eval_dataset.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "multi_weight_precision_results.json")


def load_or_seed_cache() -> dict[str, dict[str, int]]:
    """Loads judgment_cache.json, seeding it once from manual_precision_results.json if absent."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    cache: dict[str, dict[str, int]] = {}
    if os.path.exists(LEGACY_RESULTS_PATH):
        with open(LEGACY_RESULTS_PATH, "r", encoding="utf-8") as f:
            legacy_records = json.load(f)
        for row in legacy_records:
            q = row.get("query", "")
            q_map = cache.setdefault(q, {})
            for item in row.get("judgments", []):
                pmid = str(item.get("pmid", ""))
                if pmid and "relevant" in item:
                    q_map[pmid] = int(item["relevant"])
        save_cache(cache)
    return cache


def save_cache(cache: dict[str, dict[str, int]]) -> None:
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)


def run_interactive_hand_judging(
    weights_to_judge: list[float] | None = None,
    candidate_limit: int = 15,
    top_k: int = 5,
):
    if weights_to_judge is None:
        weights_to_judge = [0.3, 0.7, 0.95]

    if not os.path.exists(DATASET_PATH):
        print(f"❌ Gold dataset not found at {DATASET_PATH}. Run generate_from_config.py first.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    cache = load_or_seed_cache()
    queries = data["queries"]
    per_query_evaluations = []

    print(f"👩‍⚖️ Multi-Weight Hand-Judging Session ({len(queries)} queries, weights={weights_to_judge}, top-{top_k} union).")
    print("Instructions: Rate uncached docs as [1] (Relevant) or [0] (Irrelevant).\n" + "=" * 70)

    for idx, item in enumerate(queries, 1):
        q = item["query"]
        lang = item["lang"]
        q_cache = cache.setdefault(q, {})

        print(f"\n[{idx}/{len(queries)}] Query ({lang.upper()}): {q}")
        print("-" * 70)

        vector = embed_query(q)
        medical_terms = extract_medical_terms(q)

        search_results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=candidate_limit,
            output_fields=["abstract", "title", "lang", "pmid", "doi"],
        )

        if not search_results or not search_results[0]:
            print("⚠️ No search results returned.")
            continue

        per_weight_pmids: dict[float, list[str]] = {}
        union_docs: dict[str, dict] = {}

        for w in weights_to_judge:
            kw_weight = round(1.0 - w, 2)
            filtered = filter_results_by_relevance(
                search_results[0], q, medical_terms, semantic_weight=w, keyword_weight=kw_weight
            )
            hits = filtered[:top_k]
            pmids: list[str] = []
            for hit in hits:
                pmid = get_doc_id(hit)
                pmids.append(pmid)
                if pmid not in union_docs:
                    union_docs[pmid] = {
                        "pmid": pmid,
                        "title": getattr(hit.entity, "title", "No Title"),
                        "abstract": getattr(hit.entity, "abstract", "No Abstract"),
                        "weights": [w],
                    }
                else:
                    union_docs[pmid]["weights"].append(w)
            per_weight_pmids[w] = pmids

        cached_hits = sum(1 for pmid in union_docs if pmid in q_cache)
        print(f"  Union size: {len(union_docs)} unique docs ({cached_hits} cached, {len(union_docs) - cached_hits} to judge)")

        for pmid, doc in union_docs.items():
            if pmid in q_cache:
                continue

            abstract = doc["abstract"]
            snippet = abstract[:250] + "..." if len(abstract) > 250 else abstract
            print(f"\n  [NEW] PMID: {pmid} | Retrieved at weights: {doc['weights']}")
            print(f"  Title: {doc['title']}")
            print(f"  Snippet: {snippet}")

            while True:
                choice = input("  -> Relevant? [1/0]: ").strip()
                if choice in ("0", "1"):
                    q_cache[pmid] = int(choice)
                    save_cache(cache)
                    break
                print("  ❌ Invalid input. Please enter 1 for relevant or 0 for irrelevant.")

        weight_scores = {}
        for w in weights_to_judge:
            ranked_pmids = per_weight_pmids[w]
            p3 = precision_at_k(ranked_pmids, q_cache, 3)
            p5 = precision_at_k(ranked_pmids, q_cache, 5)
            weight_scores[str(w)] = {
                "semantic_weight": w,
                "keyword_weight": round(1.0 - w, 2),
                "top_5_pmids": ranked_pmids,
                "precision_at_3": round(p3, 4),
                "precision_at_5": round(p5, 4),
            }
            print(f"  🎯 [w={w}] P@3: {p3:.2f} | P@5: {p5:.2f}")

        print("=" * 70)
        per_query_evaluations.append({
            "query": q,
            "lang": lang,
            "union_doc_count": len(union_docs),
            "by_weight": weight_scores,
        })

    # Macro-average P@3 and P@5 per weight across all judged queries
    summary_by_weight = {}
    n_eval = len(per_query_evaluations)
    for w in weights_to_judge:
        w_key = str(w)
        mean_p3 = sum(r["by_weight"][w_key]["precision_at_3"] for r in per_query_evaluations) / n_eval if n_eval else 0.0
        mean_p5 = sum(r["by_weight"][w_key]["precision_at_5"] for r in per_query_evaluations) / n_eval if n_eval else 0.0
        summary_by_weight[w_key] = {
            "semantic_weight": w,
            "keyword_weight": round(1.0 - w, 2),
            "mean_precision_at_3": round(mean_p3, 4),
            "mean_precision_at_5": round(mean_p5, 4),
        }

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(),
        "weights_evaluated": weights_to_judge,
        "candidate_limit": candidate_limit,
        "top_k": top_k,
        "summary_by_weight": summary_by_weight,
        "results": per_query_evaluations,
    }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n🏁 Multi-Weight Hand-Judging Complete!")
    for w_key, stats in summary_by_weight.items():
        print(f"📊 Weight {w_key}: Mean P@3 = {stats['mean_precision_at_3']:.4f} | Mean P@5 = {stats['mean_precision_at_5']:.4f}")
    print(f"💾 Saved multi-weight evaluations to {OUTPUT_PATH}")
    print(f"🗂️  Updated judgment cache at {CACHE_PATH}")


if __name__ == "__main__":
    run_interactive_hand_judging()

