import sys
import os
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
from itertools import combinations

from datetime import datetime, timezone
from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance

# Import centralized queries from test_config to remain DRY
from tests.test_config import GENERAL_MEDICAL_QUERIES, INDONESIAN_MEDICAL_QUERIES, COMPLEX_QUERIES

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")

def compute_robust_statistics(raw_durations: list[float], token_counts: list[int]) -> dict:
    """Discards the first run as warmup, then computes median and IQR for duration."""
    eval_runs = raw_durations[1:] if len(raw_durations) > 1 else raw_durations
    eval_tokens = token_counts[1:] if len(token_counts) > 1 else token_counts
    
    median_lat = float(np.median(eval_runs))
    q75, q25 = np.percentile(eval_runs, [75, 25])
    iqr_lat = float(q75 - q25)
    estimated_context_tokens = float(np.mean(eval_tokens)) if eval_tokens else 0.0
    
    return {
        "median_latency_seconds": round(median_lat, 3),
        "iqr_latency_seconds": round(iqr_lat, 3),
        "estimated_context_tokens_generated": round(estimated_context_tokens, 1),
        "total_trials_recorded": len(eval_runs),
        "warmup_discarded": True
    }

def get_git_commit() -> str:
    """Returns the current HEAD commit hash, or 'unknown' if unavailable."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"

git_commit = get_git_commit()


def get_doc_id(hit) -> str:
    pmid = getattr(hit.entity, "pmid", None)
    if pmid not in (None, ""):
        return str(pmid)
    return str(getattr(hit, "id", ""))


def top_k_overlap(ids_a: list[str], ids_b: list[str]) -> dict:
    set_a, set_b = set(ids_a), set(ids_b)
    denom = len(set_a | set_b)
    return {
        "shared_count": len(set_a & set_b),
        "jaccard": round(len(set_a & set_b) / denom, 4) if denom else 1.0,
    }


def kendall_tau_union(ranking_a: list[str], ranking_b: list[str]) -> float:
    """Computes Kendall's tau over the union of two ordered doc ID lists.

    Docs missing from a list are assigned rank len(list) (worst rank in that list).
    """
    union_ids = list(dict.fromkeys(ranking_a + ranking_b))
    n = len(union_ids)
    if n < 2:
        return 1.0

    rank_a = {doc_id: idx for idx, doc_id in enumerate(ranking_a)}
    rank_b = {doc_id: idx for idx, doc_id in enumerate(ranking_b)}
    default_a, default_b = len(ranking_a), len(ranking_b)

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = rank_a.get(union_ids[i], default_a) - rank_a.get(union_ids[j], default_a)
            db = rank_b.get(union_ids[i], default_b) - rank_b.get(union_ids[j], default_b)
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    denom = concordant + discordant
    return round((concordant - discordant) / denom, 4) if denom else 1.0


def run_weight_sweep(repeats: int = 5):
    queries = (
        GENERAL_MEDICAL_QUERIES[:4] + 
        INDONESIAN_MEDICAL_QUERIES[:2] + 
        COMPLEX_QUERIES[:2]
    )
    
    weights_to_test = [0.3, 0.5, 0.7, 0.85, 0.95]
    sweep_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scoring": "normalized",  # Add scoring information
        "git_commit": git_commit,
        "queries_tested": queries,
        "repeats_per_weight": repeats,
        "results": {},
        "rank_invariance": {},
    }

    print(f"⚖️ Running Hybrid Weight Sweep across {len(queries)} queries (Repeats: {repeats}, Warmup: 1)\n")

    for q in queries:
        print(f"--- Query: '{q[:40]}...' ---")
        vector = embed_query(q)
        medical_terms = extract_medical_terms(q)
        
        q_results = []
        for w in weights_to_test:
            kw_weight = round(1.0 - w, 2)
            durations = []
            token_counts = []
            last_precision = 0.0

            for trial in range(repeats):
                start_time = time.perf_counter()
                
                results = collection.search(
                    data=[vector],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=15,
                    output_fields=["abstract", "title", "lang", "pmid", "doi"]
                )
                
                filtered = filter_results_by_relevance(results[0], q, medical_terms, semantic_weight=w, keyword_weight=kw_weight)
                top_3 = filtered[:3]
                top_3_doc_ids = [get_doc_id(hit) for hit in top_3]

                matches = 0
                for hit in top_3:
                    abstract = getattr(hit.entity, "abstract", "").lower()
                    if any(term in abstract for term in medical_terms):
                        matches += 1
                precision = matches / max(1, len(top_3))
                last_precision = precision

                end_time = time.perf_counter()
                duration = end_time - start_time
                durations.append(duration)
                
                # Proxy token estimation based on abstract length processed
                total_abstract_chars = sum(len(getattr(hit.entity, "abstract", "")) for hit in top_3)
                token_counts.append(int(total_abstract_chars / 4.0))


            stats = compute_robust_statistics(durations, token_counts)
            q_results.append({
                "semantic_weight": w,
                "keyword_weight": kw_weight,
                "precision_at_3": last_precision,
                "top_3_doc_ids": top_3_doc_ids,
                "metrics": stats
            })
            print(f"  [Weight {w}] Median Latency: {stats['median_latency_seconds']}s | IQR: {stats['iqr_latency_seconds']}s")

        pairwise_invariance = []
        for entry_a, entry_b in combinations(q_results, 2):
            ov = top_k_overlap(entry_a["top_3_doc_ids"], entry_b["top_3_doc_ids"])
            tau = kendall_tau_union(entry_a["top_3_doc_ids"], entry_b["top_3_doc_ids"])
            pairwise_invariance.append({
        "w_a": entry_a["semantic_weight"],
        "w_b": entry_b["semantic_weight"],
        "top_3_shared": ov["shared_count"],
        "top_3_jaccard": ov["jaccard"],
        "kendall_tau": tau,
            })

        sweep_data["results"][q] = q_results
        sweep_data["rank_invariance"][q] = pairwise_invariance

    output_path = os.path.join(os.path.dirname(__file__), "weight_sweep_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sweep_data, f, indent=2)
    print(f"\n✅ Weight sweep complete! Robust metrics saved to {output_path}")

if __name__ == "__main__":
    run_weight_sweep(repeats=5)

