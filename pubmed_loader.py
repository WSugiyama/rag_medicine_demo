import os
import json
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

Entrez.email = os.getenv("NCBI_EMAIL")
# Entrez.api_key = os.getenv("NCBI_API_KEY", None)

if not Entrez.email:
    raise ValueError("❌ NCBI_EMAIL is missing from your environment or .env file!")

def fetch_pubmed(query, max_results=250, batch_size=100):
    print(f"🔍 Searching PubMed for query: '{query}' (Target max: {max_results})")
    
    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        record = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"⚠️ Search error for '{query}': {e}")
        return []
    
    ids = record.get("IdList", [])
    total_ids = len(ids)
    print(f"📄 Found {total_ids} articles for '{query}'. Fetching details...")
    
    results = []
    for i in range(0, total_ids, batch_size):
        chunk_ids = ids[i:i + batch_size]
        chunk_num = (i // batch_size) + 1
        total_chunks = (total_ids + batch_size - 1) // batch_size
        
        print(f"📥 Fetching batch {chunk_num}/{total_chunks} ({len(chunk_ids)} articles)...")
        
        try:
            handle = Entrez.efetch(db="pubmed", id=chunk_ids, rettype="abstract", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for article in records.get("PubmedArticle", []):
                try:
                    medline = article["MedlineCitation"]
                    article_info = medline["Article"]
                    
                    title = article_info.get("ArticleTitle", "No Title")
                    pmid = str(medline["PMID"])
                    
                    abstract_text = ""
                    if "Abstract" in article_info and "AbstractText" in article_info["Abstract"]:
                        abstract_elements = article_info["Abstract"]["AbstractText"]
                        parts = []
                        for elem in abstract_elements:
                            label = elem.attributes.get("Label", "") if hasattr(elem, "attributes") else ""
                            text = str(elem)
                            if label:
                                parts.append(f"{label}: {text}")
                            else:
                                parts.append(text)
                        abstract_text = " ".join(parts)
                    
                    lang = article_info.get("Language", ["en"])[0]
                    
                    doi = None
                    if "PubmedData" in article and "ArticleIdList" in article["PubmedData"]:
                        for article_id in article["PubmedData"]["ArticleIdList"]:
                            if hasattr(article_id, "attributes") and article_id.attributes.get("IdType") == "doi":
                                doi = str(article_id)
                                break
                    
                    if abstract_text:
                        results.append({
                            "pmid": pmid,
                            "title": str(title),
                            "abstract": abstract_text,
                            "lang": str(lang),
                            "doi": doi
                        })
                except Exception:
                    continue
        except Exception as batch_e:
            print(f"⚠️ Error fetching batch {chunk_num}: {batch_e}")
            continue

    print(f"✅ Successfully extracted {len(results)} valid articles for query: '{query}'.")
    return results

if __name__ == "__main__":
    # Multi-domain query expansion to cover test_config.py requirements
    target_queries = [
        "integrative medicine",
        "COVID-19 treatment symptoms variants",
        "probiotics gut microbiome digestive health",
        "anti-inflammatory herbs arthritis turmeric",
        "meditation anxiety depression mental health",
        "antibiotics side effects clinical",
        "acupuncture chronic pain management",
        "omega-3 fatty acids heart health",
        "infeksi saluran pernapasan gejala",
        "faktor risiko kardiovaskular"
    ]


    output_filename = "pubmed_integrative.json"
    
    # Load existing records if they exist to avoid overwriting
    existing_data = []
    seen_pmids = set()
    if os.path.exists(output_filename):
        try:
            with open(output_filename, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                seen_pmids = {item["pmid"] for item in existing_data}
            print(f"📂 Loaded existing dataset with {len(existing_data)} records.")
        except Exception as e:
            print(f"⚠️ Could not read existing file, starting fresh: {e}")

    new_records_count = 0
    for q in target_queries:
        batch_results = fetch_pubmed(q, max_results=200, batch_size=100)
        for item in batch_results:
            if item["pmid"] not in seen_pmids:
                seen_pmids.add(item["pmid"])
                existing_data.append(item)
                new_records_count += 1

    # Save expanded dataset
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n💾 Saved expanded dataset. Added {new_records_count} new articles. Total corpus size: {len(existing_data)}")

