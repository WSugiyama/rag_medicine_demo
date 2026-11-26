import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")
collection.load()

def evaluate_retrieval(k_values=[3, 5]):
    dataset_path = os.path.join(os.path.dirname(__file__), "..", "tests", "gold_eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"❌ Gold dataset not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    queries = data["queries"]
    print(f"🎯 Evaluating Retrieval Performance across {len(queries)} naturalistic queries\n")

    results_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_queries": len(queries),
        "metrics": {f"recall@{k}": 0.0 for k in k_values},
        "mrr": 0.0,
        "per_query_results": []
    }

    evaluated_count = 0
    sum_mrr = 0.0
    sum_recalls = {k: 0.0 for k in k_values}
       
    for item in queries:
        q = item["query"]
        gold_pmids = set(str(p) for p in item["gold_pmids"])
        
        # Skip queries with no ground truth in the local corpus
        if not gold_pmids:
            continue

        evaluated_count += 1
 
        vector = embed_query(q)
        medical_terms = extract_medical_terms(q)

        # Search Milvus
        search_results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=max(k_values) * 3,
            output_fields=["abstract", "title", "lang", "pmid", "doi"]
        )

        if not search_results or not search_results[0]:
            print(f"⚠️ No results for: '{q[:40]}...'")
            continue

        # Apply hybrid filter
        filtered = filter_results_by_relevance(search_results[0], q, medical_terms, semantic_weight=0.3, keyword_weight=0.7)

        # Calculate Reciprocal Rank (MRR)
        first_hit_rank = None
        retrieved_pmids = []
        for rank, hit in enumerate(filtered, 1):
            pmid = str(getattr(hit.entity, "pmid", ""))
            retrieved_pmids.append(pmid)
            if pmid in gold_pmids and first_hit_rank is None:
                first_hit_rank = rank

        mrr_score = (1.0 / first_hit_rank) if first_hit_rank else 0.0
        sum_mrr += mrr_score

        # Calculate Recall@K
        query_recalls = {}
        for k in k_values:
            top_k_pmids = set(retrieved_pmids[:k])
            hits = len(top_k_pmids.intersection(gold_pmids))
            recall = hits / len(gold_pmids) if gold_pmids else 0.0
            sum_recalls[k] += recall
            query_recalls[f"recall@{k}"] = recall

        results_summary["per_query_results"].append({
            "query": q,
            "mrr": mrr_score,
            **query_recalls
        })

    # Averages
    if evaluated_count > 0:
        results_summary["mrr"] = round(sum_mrr / evaluated_count, 4)
        for k in k_values:
            results_summary["metrics"][f"recall@{k}"] = round(sum_recalls[k] / evaluated_count, 4)
    else:
        print("⚠️ Warning: 0 queries were evaluated! Check your gold dataset mapping.")

    results_summary["evaluated_queries_count"] = evaluated_count

    """
    n = len(queries)
    results_summary["mrr"] = round(sum_mrr / n, 4)
    for k in k_values:
        results_summary["metrics"][f"recall@{k}"] = round(sum_recalls[k] / n, 4)
    """

    # Save results
    output_path = os.path.join(os.path.dirname(__file__), "gold_evaluation_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print("📊 Evaluation Summary:")
    print(f"  • MRR: {results_summary['mrr']}")
    for k in k_values:
        print(f"  • Recall@{k}: {results_summary['metrics'][f'recall@{k}']}")
    print(f"\n✅ Detailed results saved to {output_path}")

if __name__ == "__main__":
    evaluate_retrieval()

