import time
import json
from datetime import datetime, timezone
from rag_pubmed_assistant import generate_answer_with_rag

# A standard test query representing a typical medical research question
TEST_QUERY = "What are the clinical benefits and mechanisms of omega-3 fatty acids in cardiovascular health?"

def run_benchmark():
    k_values = [3, 5, 10]
    benchmark_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": TEST_QUERY,
        "ablation_type": "top_k_retrieval",
        "results": []
    }

    print(f"🔬 Running Top-K Ablation Benchmark for query: '{TEST_QUERY}'\n")

    for k in k_values:
        print(f"--- Testing K = {k} ---")
        start_time = time.perf_counter()
        
        # Call your existing function with the specific top_k
        output = generate_answer_with_rag(TEST_QUERY, top_k=k)
        
        end_time = time.perf_counter()
        duration = end_time - start_time
        
        benchmark_data["results"].append({
            "top_k": k,
            "status": output["status"],
            "latency_seconds": round(duration, 2),
            "sources_count": len(output["sources"])
        })
        print(f"⏱️ K={k} completed in {duration:.2f}s\n")

    # Save results to the golden JSON artifact tracked in Git
    output_filename = "benchmark_results.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"✅ Benchmark complete! Metrics successfully written to {output_filename}")

if __name__ == "__main__":
    run_benchmark()

