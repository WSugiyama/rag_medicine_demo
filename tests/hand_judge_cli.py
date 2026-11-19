import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")
collection.load()

def run_interactive_hand_judging():
    dataset_path = os.path.join(os.path.dirname(__file__), "gold_eval_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"❌ Gold dataset not found at {dataset_path}. Run generate_from_config.py first.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    queries = data["queries"]
    judging_results = []

    print(f"👩‍⚖️ Starting Hand-Judging Session for {len(queries)} queries (Top-5 docs each).")
    print("Instructions: Review the Title & Abstract snippet and rate relevance as [1] (Relevant) or [0] (Irrelevant).\n" + "="*70)

    for idx, item in enumerate(queries, 1):
        q = item["query"]
        lang = item["lang"]
        
        print(f"\n[{idx}/{len(queries)}] Query ({lang.upper()}): {q}")
        print("-" * 70)

        vector = embed_query(q)
        medical_terms = extract_medical_terms(q)

        search_results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=5,
            output_fields=["abstract", "title", "lang", "pmid", "doi"]
        )

        if not search_results or not search_results[0]:
            print("⚠️ No search results returned.")
            continue

        filtered = filter_results_by_relevance(search_results[0], q, medical_terms, semantic_weight=0.7, keyword_weight=0.3)
        
        query_judgments = []
        for rank, hit in enumerate(filtered[:5], 1):
            pmid = str(getattr(hit.entity, "pmid", ""))
            title = getattr(hit.entity, "title", "No Title")
            abstract = getattr(hit.entity, "abstract", "No Abstract")
            snippet = abstract[:250] + "..." if len(abstract) > 250 else abstract

            print(f"\n  Rank {rank} | PMID: {pmid}")
            print(f"  Title: {title}")
            print(f"  Snippet: {snippet}")
            
            while True:
                choice = input("  -> Relevant? [1/0]: ").strip()
                if choice in ["0", "1"]:
                    query_judgments.append({"pmid": pmid, "rank": rank, "relevant": int(choice)})
                    break
                print("  ❌ Invalid input. Please enter 1 for relevant or 0 for irrelevant.")

        # Compute Precision@5 for this query
        relevant_count = sum(j["relevant"] for j in query_judgments)
        p_at_5 = relevant_count / len(query_judgments) if query_judgments else 0.0
        print(f"\n  🎯 Query Precision@5: {p_at_5:.2f} ({relevant_count}/5)")
        print("=" * 70)

        judging_results.append({
            "query": q,
            "lang": lang,
            "precision_at_5": p_at_5,
            "judgments": query_judgments
        })

    # Save final manual judgments
    output_path = os.path.join(os.path.dirname(__file__), "manual_precision_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(judging_results, f, indent=2, ensure_ascii=False)

    # Compute macro-average P@5 across all judged queries
    mean_p5 = sum(r["precision_at_5"] for r in judging_results) / len(judging_results) if judging_results else 0.0
    print(f"\n🏁 Hand-Judging Complete!")
    print(f"📊 Overall Mean Hand-Judged Precision@5: {mean_p5:.4f}")
    print(f"💾 Saved structured evaluations to {output_path}")

if __name__ == "__main__":
    run_interactive_hand_judging()

