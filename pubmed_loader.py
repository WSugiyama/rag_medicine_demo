from Bio import Entrez
import json

# Set email and optional API key
Entrez.email = "REDACTED@example.com"
# Entrez.api_key = "your_ncbi_api_key"  # Optional

def fetch_pubmed(query, max_results=1024):
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    ids = record["IdList"]
    
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
    records = Entrez.read(handle)
    
    results = []
    for article in records["PubmedArticle"]:
        try:
            title = article["MedlineCitation"]["Article"]["ArticleTitle"]
            abstract = article["MedlineCitation"]["Article"]["Abstract"]["AbstractText"][0]
            lang = article["MedlineCitation"]["Article"]["Language"][0]
            pmid = article["MedlineCitation"]["PMID"]
            results.append({
                "pmid": str(pmid),
                "title": title,
                "abstract": abstract,
                "lang": lang
            })
        except Exception:
            continue  # Some articles might not have abstracts
    
    return results

# Example usage:
data = fetch_pubmed("integrative medicine", max_results=128)

# Save if needed
with open("pubmed_integrative.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

# Preview
for item in data:
    print(f"\n🧠 {item['title']} [{item['lang']}]\n{item['abstract'][:200]}...")

