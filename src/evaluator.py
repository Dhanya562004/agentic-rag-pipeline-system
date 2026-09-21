"""
Evaluation Layer Module
Basic confidence scoring and relevance check without using external models.
"""

def evaluate_response(query: str, answer: str, mode: str, retrieved_chunks: list = None) -> dict:
    """
    Evaluate response quality and assign confidence score (High / Medium / Low).
    
    Args:
        query: Original user query
        answer: Generated LLM response
        mode: Pipeline execution mode ('rag' or 'general')
        retrieved_chunks: Chunks retrieved during RAG phase
        
    Returns:
        Dict containing confidence rating and relevance score
    """
    # Guard clause for empty or error responses
    if not answer or answer.strip() == "" or "Error" in answer or "No relevant" in answer:
        return {
            "confidence": "Low",
            "relevance_score": 0.0
        }
    
    # Calculate simple word overlap between query and answer (excluding common stop words)
    stop_words = {"what", "is", "the", "a", "an", "of", "in", "and", "to", "for", "where", "how", "who", "which", "tell", "me", "about"}
    query_words = set(query.lower().split()) - stop_words
    answer_words = set(answer.lower().split()) - stop_words
    
    if not query_words:
        overlap_ratio = 1.0
    else:
        overlap = query_words.intersection(answer_words)
        overlap_ratio = len(overlap) / len(query_words)

    # Determine confidence score based on retrieval status, word count, and overlap
    if mode == "rag":
        has_retrieval = bool(retrieved_chunks and len(retrieved_chunks) > 0)
        if has_retrieval and (overlap_ratio > 0.25 or len(answer.split()) > 10):
            confidence = "High"
        elif has_retrieval:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        # General LLM mode confidence heuristic
        if len(answer.split()) >= 8:
            confidence = "High" if overlap_ratio > 0.2 else "Medium"
        else:
            confidence = "Low"
            
    return {
        "confidence": confidence,
        "relevance_score": round(overlap_ratio, 2)
    }
