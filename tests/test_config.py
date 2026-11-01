"""
Configuration file for RAG system testing
"""

# Test Categories and Sample Queries
COVID_QUERIES = [
    "What are the main symptoms of COVID-19?",
    "How effective are COVID-19 vaccines against new variants?",
    "Can long COVID be treated with probiotics?",
    "What are the neurological effects of SARS-CoV-2?",
    "How does COVID-19 affect the cardiovascular system?",
    "Are there natural treatments for COVID-19 symptoms?",
    "What is the effectiveness of antivirals for COVID-19?",
    "How do COVID-19 symptoms differ from flu symptoms?",
    "What are the risk factors for severe COVID-19?",
    "Can herbal medicines help with COVID-19 recovery?"
]

GENERAL_MEDICAL_QUERIES = [
    "What are the benefits of probiotics for digestive health?",
    "How do anti-inflammatory herbs help with arthritis?",
    "What natural treatments are effective for anxiety?",
    "How does meditation affect immune system function?",
    "What are the side effects of common antibiotics?",
    "Can acupuncture help with chronic pain management?",
    "What dietary supplements support heart health?",
    "How effective is turmeric for inflammation?",
    "What are the benefits of omega-3 fatty acids?",
    "Can yoga help with depression symptoms?"
]

INDONESIAN_MEDICAL_QUERIES = [
    "Apa saja gejala utama dari infeksi COVID-19?",
    "Bagaimana efektivitas vaksin terhadap varian baru?",
    "Apakah probiotik dapat membantu pemulihan pasien?",
    "Apa saja faktor risiko untuk penyakit kardiovaskular?"
]

MULTILINGUAL_QUERIES = [
    ("¿Cuáles son los efectos del COVID-19 en el sistema inmune?", "es"),
    ("Comment les probiotiques affectent-ils la santé digestive?", "fr"),
    ("Was sind die Vorteile von Meditation für die Gesundheit?", "de"),
    ("Quali sono i sintomi principali del COVID-19?", "it"),
    ("COVID-19ワクチンの効果はどのようなものですか？", "ja")
]

COMPLEX_QUERIES = [
    "How do probiotics interact with COVID-19 vaccines and what are the implications for immune response?",
    "Can traditional Chinese medicine approaches combined with modern antivirals improve COVID-19 outcomes?",
    "What is the relationship between gut microbiome, probiotics, and COVID-19 severity?",
    "How do anti-inflammatory compounds in turmeric compare to conventional COVID-19 treatments?",
    "What are the long-term neurological effects of COVID-19 and can they be mitigated with natural interventions?"
]

# Expected Performance Benchmarks
PERFORMANCE_BENCHMARKS = {
    "term_extraction_accuracy": 0.8,  # 80% accuracy in extracting medical terms
    "citation_coverage": 0.9,         # 90% of responses should have citations
    "response_relevance": 0.85,       # 85% of responses should be relevant
    "multilingual_support": 0.7,      # 70% success rate for non-English queries
    "complex_query_handling": 0.6     # 60% success rate for complex queries
}

# Test Data for Mocking
MOCK_ABSTRACTS = {
    "covid": [
        "COVID-19 is caused by SARS-CoV-2 coronavirus and primarily affects respiratory system causing symptoms like fever, cough, and shortness of breath.",
        "Clinical trials demonstrate COVID-19 vaccines are highly effective in preventing severe disease and hospitalization.",
        "Long COVID symptoms can persist for months and may include fatigue, brain fog, and breathing difficulties.",
        "SARS-CoV-2 can cause neurological complications including loss of taste and smell, headaches, and cognitive impairment."
    ],
    "probiotics": [
        "Probiotics are live microorganisms that provide health benefits when consumed in adequate amounts, particularly for digestive health.",
        "Clinical studies show probiotics can help restore gut microbiome balance and improve symptoms of irritable bowel syndrome.",
        "Lactobacillus and Bifidobacterium strains are commonly used probiotics with demonstrated benefits for immune function."
    ],
    "anti_inflammatory": [
        "Turmeric contains curcumin, a potent anti-inflammatory compound that may help reduce inflammation markers.",
        "Omega-3 fatty acids have anti-inflammatory properties and may help reduce risk of cardiovascular disease.",
        "Green tea polyphenols exhibit anti-inflammatory effects and may support overall health."
    ]
}

# Citation Patterns for Testing
CITATION_PATTERNS = [
    r'\[\d+\]',           # Basic citation pattern [1], [2], etc.
    r'\[\d+,\s*\d+\]',  # Multiple citations [1, 2]
    r'\[\d+-\d+\]',      # Range citations [1-3]
]

# Expected Medical Terms by Category
EXPECTED_TERMS = {
    "covid": ["covid", "coronavirus", "sars-cov"],
    "probiotics": ["probiotic", "microbiome", "lactobacillus", "bifidobacterium"],
    "inflammation": ["anti-inflammatory", "inflammation", "inflammatory"],
    "treatment": ["treatment", "therapy", "intervention", "clinical"],
    "natural": ["natural", "herbal", "traditional", "alternative"]
}

# Error Patterns to Check For
ERROR_PATTERNS = [
    "No results found",
    "No relevant results found",
    "Error:",
    "Exception:",
    "Failed to"
]

# Quality Metrics Configuration
QUALITY_METRICS = {
    "min_response_length": 50,      # Minimum characters in response
    "max_response_length": 2000,    # Maximum characters in response
    "min_sources": 1,               # Minimum number of sources
    "max_sources": 5,               # Maximum number of sources
    "required_citation_ratio": 0.3  # Min ratio of sentences with citations
}


