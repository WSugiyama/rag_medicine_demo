import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pymilvus import connections, Collection
from rag_pubmed_assistant import extract_medical_terms

connections.connect(alias="default", host="localhost", port="19530")
collection = Collection(name="PubMed_Integrative_v2")
collection.load()

# Self-contained evaluation query pool (decoupled from test_config.py to protect benchmarks)
EVAL_QUERIES = [
    {"query": "How effective is acupuncture for chronic pain management?", "lang": "en"},
    {"query": "What are the benefits of probiotics for digestive health?", "lang": "en"},
    {"query": "How do anti-inflammatory herbs help with arthritis?", "lang": "en"},
    {"query": "What natural treatments are effective for anxiety?", "lang": "en"},
    {"query": "How does meditation affect immune system function?", "lang": "en"},
    {"query": "Can yoga help with depression symptoms?", "lang": "en"},
    {"query": "What dietary supplements support heart health?", "lang": "en"},
    {"query": "How effective is turmeric for inflammation?", "lang": "en"},
    {"query": "What are the health benefits of omega-3 fatty acids?", "lang": "en"},
    {"query": "Are green tea extracts safe for long-term liver health?", "lang": "en"},
    {"query": "Apa saja manfaat terapi relaksasi bagi penderita kecemasan kronis?", "lang": "id"},
    {"query": "Bagaimana efektivitas pengobatan herbal terhadap nyeri sendi?", "lang": "id"}
]

def generate_gold_dataset_from_config():
    print(f"🔍 Scanning Milvus corpus for {len(EVAL_QUERIES)} decoupled evaluation queries...")

    # Fetch records from Milvus to search through
    records = collection.query(expr="pmid != ''", output_fields=["pmid", "abstract", "title"], limit=2000)

    gold_dataset = {
        "description": "Self-contained ground-truth evaluation set decoupled from test_config.py.",
        "queries": []
    }

    matched_count = 0
    for item in EVAL_QUERIES:
        q = item["query"]
        lang = item["lang"]
        medical_terms = extract_medical_terms(q)
        
        matched_pmids = []
        for r in records:
            abstract = (r.get("abstract") or "").lower()
            title = (r.get("title") or "").lower()
            pmid = str(r.get("pmid"))
            
            hits = sum(1 for term in medical_terms if term in abstract or term in title)
            if hits >= 1 and pmid not in matched_pmids:
                matched_pmids.append(pmid)
        
        gold_pmids = matched_pmids[:3]  # Max 3 gold PMIDs per query
        if gold_pmids:
            matched_count += 1

        gold_dataset["queries"].append({
            "query": q,
            "lang": lang,
            "gold_pmids": gold_pmids
        })

    # Save output
    output_path = os.path.join(os.path.dirname(__file__), "..", "tests", "gold_eval_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gold_dataset, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Successfully generated gold dataset! {matched_count}/{len(EVAL_QUERIES)} queries matched local gold PMIDs.")
    print(f"💾 Saved to {output_path}")

if __name__ == "__main__":
    generate_gold_dataset_from_config()

