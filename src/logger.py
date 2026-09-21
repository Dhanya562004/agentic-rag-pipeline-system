"""
Observability Logger Module
Logs query details, response time, execution mode, and errors using print and time module.
"""

import time

def log_execution(query: str, mode: str, response_time: float, confidence: str, error: str = None):
    """
    Log execution details to standard output for basic observability.
    
    Args:
        query: User query
        mode: Pipeline mode ('rag' or 'general')
        response_time: Total execution time in seconds
        confidence: Evaluated confidence score ('High' / 'Medium' / 'Low')
        error: Error message string if an exception occurred
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n--- [PIPELINE LOG | {timestamp}] ---")
    print(f"Query:         {query}")
    print(f"Mode:          {mode}")
    print(f"Response Time: {response_time:.3f} seconds")
    print(f"Confidence:    {confidence}")
    if error:
        print(f"Status:        ERROR -> {error}")
    else:
        print(f"Status:        SUCCESS")
    print("-------------------------------------\n")
