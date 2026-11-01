"""
Empirical Weight Sweep script for Hybrid Retrieval (Semantic vs. Keyword Weighting)
"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pymilvus import connections, Collection
from rag_pubmed_assistant import embed_query, extract_medical_terms, filter_results_by_relevance

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")

def run_weight_sweep():
    queries = [
        "What herbal treatments help with inflammation?",
        "How effective are COVID-19 vaccines against new variants?",
        "Can long COVID be treated with probiotics?",
        "What are the cardiovascular effects of omega-3 fatty acids?",
        "Bagaimana efektivitas akupunktur untuk meredakan nyeri kronis?", # Indonesian acupuncture query
        "What are the molecular targets of turmeric in cancer therapy?",   # Specific chemical target query
        "Can vitamin D supplementation reduce respiratory infections?",    # Supplement efficacy query
        "What mechanisms link gut microbiota to metabolic syndrome?"       # Complex multi-hop query
    ]
    
    weights_to_test = [0.3, 0.5, 0.7, 0.85, 0.95]
    sweep_results = {}
    
    for q in queries:
        vector = embed_query(q)
        medical_terms = extract_medical_terms(q)
        
        results = collection.search(
            data=[vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=15,
            output_fields=["abstract", "title", "lang", "pmid", "doi"]
        )
        
        q_results = []
        for w in weights_to_test:
            kw_weight = round(1.0 - w, 2)
            filtered = filter_results_by_relevance(results[0], q, medical_terms, semantic_weight=w, keyword_weight=kw_weight)
            top_3 = filtered[:3]
            
            matches = 0
            for hit in top_3:
                abstract = getattr(hit.entity, "abstract", "").lower()
                if any(term in abstract for term in medical_terms):
                    matches += 1
            precision = matches / max(1, len(top_3))
            q_results.append({"semantic_weight": w, "keyword_weight": kw_weight, "precision_at_3": precision})
            
        sweep_results[q] = q_results
        
    # Save inside the benchmarks directory for neat organization
    output_path = os.path.join(os.path.dirname(__file__), "weight_sweep_results.json")
    with open(output_path, "w") as f:
        json.dump(sweep_results, f, indent=2)
    print(f"Weight sweep completed successfully and saved to {output_path}")

if __name__ == "__main__":
    run_weight_sweep()

