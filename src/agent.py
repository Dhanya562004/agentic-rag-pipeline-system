"""
Agent Router Module
Rule-based intent classifier that routes queries to either:
1. 'rag' (Document Retrieval mode)
2. 'general' (General Knowledge LLM mode)
"""

# Keywords related to the document domain (France geography, climate, economy, history dataset)
DOCUMENT_KEYWORDS = [
    "france", "paris", "french", "capital", "population", "gdp", 
    "geography", "history", "president", "economy", "climate", 
    "culture", "tourism", "tourist", "eiffel", "louvre", "macron",
    "europe", "currency", "euro", "language", "border", "region",
    "rainfall", "soil", "river", "plateau", "landform", "massif",
    "basin", "lowland", "agriculture", "climate", "mountain", "alps"
]

def router(query: str) -> str:
    """
    Classify query intent based on keywords and heuristics.
    
    Args:
        query: User input query string
        
    Returns:
        str: 'rag' for document-based queries, 'general' for general knowledge queries
    """
    if not query or not query.strip():
        return "general"
        
    query_lower = query.lower()
    
    # Rule-based matching: Check if any document domain keyword exists in query
    for keyword in DOCUMENT_KEYWORDS:
        if keyword in query_lower:
            return "rag"
            
    # Default fallback to general knowledge LLM mode
    return "general"

def route_query(query: str) -> str:
    """
    Alias for router() for backward compatibility with existing code.
    """
    return router(query)

