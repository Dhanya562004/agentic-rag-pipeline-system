"""
Observability Logger Module
Logs query details, execution mode, confidence rating, similarity score, response time, and status.
"""

import time

def log_execution(
    query: str, 
    mode: str, 
    response_time: float, 
    confidence: str, 
    score: float = 0.0,
    error: str = None
):
    """
    Log execution details to standard output for basic observability and monitoring.
    
    Args:
        query: User query string
        mode: Pipeline mode ('RAG', 'General LLM', or 'RAG -> General LLM (Fallback)')
        response_time: Total execution time in seconds
        confidence: Evaluated confidence rating ('High' / 'Medium' / 'Low')
        score: Numerical similarity / confidence score (0.0 to 1.0)
        error: Error message string if an exception occurred
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- [PIPELINE LOG | {timestamp}] ---")
    print(f"Query:         {query}")
    print(f"Mode:          {mode}")
    print(f"Confidence:    {confidence} (Score: {score:.2f})")
    print(f"Response Time: {response_time:.3f} seconds")
    if error:
        print(f"Status:        ERROR -> {error}")
    else:
        print(f"Status:        SUCCESS")
    print("-------------------------------------\n")

