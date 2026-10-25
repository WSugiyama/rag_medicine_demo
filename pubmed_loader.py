
import os
import json
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()

# Set email and optional API key
Entrez.email = os.getenv("NCBI_EMAIL")
# Entrez.api_key = os.getenv("NCBI_API_KEY", None)

# Check to make sure it loaded properly
if not Entrez.email:
    raise ValueError("❌ NCBI_EMAIL is missing from your environment or .env file!")


def fetch_pubmed(query, max_results=1024, batch_size=200):
    print(f"🔍 Searching PubMed for query: '{query}' (Target max: {max_results})")
    
    # 1. Search for all matching IDs
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    
    ids = record["IdList"]
    total_ids = len(ids)
    print(f"📄 Found {total_ids} total articles matching query. Fetching details in chunks of {batch_size}...")
    
    results = []

    # 2. Fetch records in chunks to prevent NCBI payload/URL length limits
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
                    pmid = medline["PMID"]
                    
                    # Handle structured abstracts (e.g., Background, Methods, Results, Conclusions)
                    abstract_text = ""
                    if "Abstract" in article_info and "AbstractText" in article_info["Abstract"]:
                        abstract_elements = article_info["Abstract"]["AbstractText"]
                        # Some elements might have labels (e.g., Label='BACKGROUND')
                        parts = []
                        for elem in abstract_elements:
                            label = elem.attributes.get("Label", "")
                            text = str(elem)
                            if label:
                                parts.append(f"{label}: {text}")
                            else:
                                parts.append(text)
                        abstract_text = " ".join(parts)
                    
                    # Extract language
                    lang = article_info.get("Language", ["en"])[0]
                    
                    # Extract DOI if available
                    doi = None
                    if "PubmedData" in article and "ArticleIdList" in article["PubmedData"]:
                        for article_id in article["PubmedData"]["ArticleIdList"]:
                            if article_id.attributes.get("IdType") == "doi":
                                doi = str(article_id)
                                break
                    
                    if abstract_text:  # Only add if an abstract actually exists
                        results.append({
                            "pmid": str(pmid),
                            "title": str(title),
                            "abstract": abstract_text,
                            "lang": str(lang),
                            "doi": doi
                        })
                except Exception as inner_e:
                    # Skip malformed individual articles without crashing the batch
                    continue
                    
        except Exception as batch_e:
            print(f"⚠️ Error fetching batch {chunk_num}: {batch_e}")
            continue

    print(f"✅ Successfully extracted {len(results)} valid articles with abstracts.")
    return results

if __name__ == "__main__":
    # Example usage for large dataset generation
    data = fetch_pubmed("integrative medicine", max_results=512, batch_size=200)

    # Save to JSON
    output_filename = "pubmed_integrative.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved dataset to {output_filename}")

    # Preview first few items
    for item in data[:3]:
        print(f"\n🧠 {item['title']} [{item['lang']}] (PMID: {item['pmid']})\n{item['abstract'][:200]}...")

