"""
Agent Router Module
Simple rule-based router to decide whether to use RAG (document retrieval) or general LLM.
"""

# Keywords related to the document domain (France dataset)
DOCUMENT_KEYWORDS = [
    "france", "paris", "french", "capital", "population", "gdp", 
    "geography", "history", "president", "economy", "climate", 
    "culture", "tourism", "tourist", "eiffel", "louvre", "macron",
    "europe", "currency", "euro", "language", "border", "region"
]

def route_query(query: str) -> str:
    """
    Decide execution path based on query keywords.
    Returns 'rag' if query relates to document domain, else 'general'.
    """
    query_lower = query.lower()
    
    # Check if any document-specific keyword is present in the query
    for keyword in DOCUMENT_KEYWORDS:
        if keyword in query_lower:
            return "rag"
            
    # Fallback to general LLM for general knowledge / off-topic queries
    return "general"
