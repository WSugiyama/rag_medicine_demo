import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import json
import numpy as np
from datetime import datetime, timezone
from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance, generate_answer_with_rag
from tests.test_config import GENERAL_MEDICAL_QUERIES, INDONESIAN_MEDICAL_QUERIES, COMPLEX_QUERIES

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")

def compute_split_statistics(retrieval_durations: list[float], generation_durations: list[float], token_counts: list[int]) -> dict:
    eval_ret = retrieval_durations[1:] if len(retrieval_durations) > 1 else retrieval_durations
    eval_gen = generation_durations[1:] if len(generation_durations) > 1 else generation_durations
    eval_tok = token_counts[1:] if len(token_counts) > 1 else token_counts
    
    return {
        "retrieval": {
            "median_seconds": round(float(np.median(eval_ret)), 4),
            "iqr_seconds": round(float(np.percentile(eval_ret, 75) - np.percentile(eval_ret, 25)), 4)
        },
        "generation": {
            "median_seconds": round(float(np.median(eval_gen)), 3),
            "iqr_seconds": round(float(np.percentile(eval_gen, 75) - np.percentile(eval_gen, 25)), 3)
        },
        "avg_tokens_generated": round(float(np.mean(eval_tok)) if eval_tok else 0.0, 1),
        "total_trials_recorded": len(eval_ret),
        "warmup_discarded": True
    }

def run_benchmark(repeats: int = 5, mode: str = "full"):
    k_values = [3, 5, 10]
    
    # Stratified sampling: Use a tiny subset for quick testing, or full suite for final reporting
    if mode == "quick":
        test_queries = [
            GENERAL_MEDICAL_QUERIES[0],      # Standard English
            INDONESIAN_MEDICAL_QUERIES[0]    # Multilingual
        ]
        print("⚡ [QUICK MODE] Running on 2 stratified queries to save GPU cycles.\n")
    else:
        test_queries = (
            GENERAL_MEDICAL_QUERIES[:2] + 
            INDONESIAN_MEDICAL_QUERIES[:1] + 
            COMPLEX_QUERIES[:1]
        )
        print(f"🔬 [FULL MODE] Running on {len(test_queries)} stratified queries.\n")

    benchmark_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "queries_tested": test_queries,
        "ablation_type": "top_k_retrieval_split",
        "repeats_per_query_per_k": repeats,
        "results": []
    }

    for k in k_values:
        print(f"--- Testing K = {k} ---")
        k_ret_durations = []
        k_gen_durations = []
        k_tokens = []
        last_status = "success"
        sources_count = 0
        
        for q_idx, query in enumerate(test_queries):
            print(f"  Query {q_idx + 1}/{len(test_queries)}: '{query[:40]}...'")
            vector = embed_query(query)
            medical_terms = extract_medical_terms(query)

            for trial in range(repeats):
                # 1. End-to-End Retrieval Timing (Embedding + Milvus Search + Filtering)
                t_ret_start = time.perf_counter()
                vector = embed_query(query)
                medical_terms = extract_medical_terms(query)
                results = collection.search(
                    data=[vector],
                    anns_field="embedding",
                    param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                    limit=k,
                    output_fields=["abstract", "title", "lang", "pmid", "doi"]
                )
                filtered = filter_results_by_relevance(results[0], query, medical_terms, semantic_weight=0.7, keyword_weight=0.3)
                t_ret_end = time.perf_counter()
                k_ret_durations.append(t_ret_end - t_ret_start)
                

                # 2. Generation Timing (Seconds via Local LLM)
                t_gen_start = time.perf_counter()
                output = generate_answer_with_rag(query, top_k=k)
                t_gen_end = time.perf_counter()
                k_gen_durations.append(t_gen_end - t_gen_start)
                
                answer_text = output.get("answer", "")
                k_tokens.append(int(len(answer_text.split()) * 1.3))
                
                last_status = output.get("status", "success")
                sources_count = len(output.get("sources", []))

        stats = compute_split_statistics(k_ret_durations, k_gen_durations, k_tokens)
        benchmark_data["results"].append({
            "top_k": k,
            "status": last_status,
            "sources_count": sources_count,
            "metrics": stats
        })
        print(f"⏱️ K={k} Summary -> Retrieval: {stats['retrieval']['median_seconds']}s | Generation: {stats['generation']['median_seconds']}s\n")

    output_filename = os.path.join(os.path.dirname(__file__), "benchmark_results_split.json")
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"✅ Benchmark complete! Metrics written to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG Top-K Benchmarks")
    parser.add_argument("--mode", type=str, default="full", choices=["quick", "full"], help="Execution mode: quick (few queries) or full suite")
    parser.add_argument("--repeats", type=int, default=3, help="Number of trial repeats per query")
    args = parser.parse_args()
    
    run_benchmark(repeats=args.repeats, mode=args.mode)

