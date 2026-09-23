"""
Evaluation Layer Module
Evaluates confidence score and relevance for responses.
Calculates confidence based on retrieval similarity score (< 0.5 threshold) and answer metrics.
"""

def confidence_evaluator(
    query: str, 
    answer: str, 
    mode: str, 
    retrieved_chunks: list = None,
    top_score: float = 0.0
) -> dict:
    """
    Evaluates response quality and assigns confidence score (High / Medium / Low)
    along with a numerical confidence rating (0.0 to 1.0).
    
    Args:
        query: User prompt
        answer: Generated text answer
        mode: Mode used ('rag', 'general', or fallback)
        retrieved_chunks: List of document chunks retrieved
        top_score: Highest similarity score from vector search
        
    Returns:
        dict: {"confidence": "High"|"Medium"|"Low", "score": float}
    """
    # Guard clause for empty or invalid answers
    if not answer or not str(answer).strip() or "Error" in str(answer):
        return {
            "confidence": "Low",
            "score": 0.0
        }
        
    invalid_phrases = ["no relevant document found", "i don't have information", "no context found"]
    if any(phrase in str(answer).lower() for phrase in invalid_phrases):
        return {
            "confidence": "Low",
            "score": round(top_score, 2)
        }

    # Evaluate based on mode and top similarity score
    if "rag" in mode.lower() and "fallback" not in mode.lower():
        num_chunks = len(retrieved_chunks) if retrieved_chunks else 0
        
        # Threshold rule: < 0.5 or 0 chunks triggers LOW confidence
        if num_chunks == 0 or top_score < 0.5:
            confidence = "Low"
            score = round(top_score, 2)
        elif top_score >= 0.7:
            confidence = "High"
            score = round(top_score, 2)
        else:
            confidence = "Medium"
            score = round(top_score, 2)
    else:
        # General LLM or Fallback mode evaluation
        word_count = len(str(answer).split())
        if word_count >= 15:
            confidence = "High"
            score = 0.85
        elif word_count >= 5:
            confidence = "Medium"
            score = 0.65
        else:
            confidence = "Low"
            score = 0.40

    return {
        "confidence": confidence,
        "score": score
    }

def evaluate_response(query: str, answer: str, mode: str, retrieved_chunks: list = None, top_score: float = 0.0) -> dict:
    """
    Alias for confidence_evaluator() for backward compatibility.
    """
    return confidence_evaluator(
        query=query, 
        answer=answer, 
        mode=mode, 
        retrieved_chunks=retrieved_chunks, 
        top_score=top_score
    )

