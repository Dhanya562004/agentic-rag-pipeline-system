"""
Pipeline Layer Module
Orchestrates Agent routing, Document Retrieval, LLM Response Generation, Evaluation, and Logging.
"""

import time
import os
import sys

# Ensure src directory is in sys.path
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from agent import route_query
from evaluator import evaluate_response
from logger import log_execution

def run_pipeline(query: str, retriever=None, llm_service=None) -> dict:
    """
    Executes the complete end-to-end AI pipeline for a given user query.
    
    Steps:
    1. Agent routes query to 'rag' or 'general' mode.
    2. If 'rag', performs hybrid vector retrieval from documents.
    3. LLM generates answer using retrieved context or general knowledge.
    4. Evaluator calculates confidence score ('High' / 'Medium' / 'Low').
    5. Logger records performance metrics and status.
    6. Returns structured output JSON format.
    
    Args:
        query: User input prompt
        retriever: Instance of VectorRetriever (optional)
        llm_service: Instance of LLMService (optional)
        
    Returns:
        dict: {"mode": "rag" | "general", "answer": str, "confidence": "High" | "Medium" | "Low"}
    """
    start_time = time.time()
    error_msg = None
    retrieved_chunks = []
    answer = ""
    
    try:
        # Step 1: Agent decides flow mode (rule-based keyword routing)
        mode = route_query(query)
        
        # Step 2 & 3: Perform Retrieval (if RAG) & Generate LLM Answer
        if mode == "rag":
            # Retrieve relevant document chunks
            if retriever:
                retrieved_chunks = retriever.hybrid_retrieve(query=query, top_k=5)
            
            # Format context from retrieved chunks
            if retrieved_chunks:
                context_parts = [
                    f"[Source {i+1}: {chunk.get('title', 'Doc')} - {chunk.get('category', 'General')}]\n{chunk.get('text', '')}"
                    for i, chunk in enumerate(retrieved_chunks)
                ]
                context = "\n\n".join(context_parts)
            else:
                context = ""
                
            # Generate response via LLM service using context
            if llm_service:
                if context:
                    answer = llm_service.generate_response(query=query, context=context, max_tokens=500)
                else:
                    answer = "No relevant document information found in the knowledge base."
            else:
                answer = f"LLM service unavailable. Retrieved context:\n{context[:400]}..." if context else "No context found."
        else:
            # General LLM mode (no retrieval required)
            if llm_service:
                answer = llm_service.generate_general_response(query=query, max_tokens=500)
            else:
                answer = f"General response: This query ('{query}') was routed to general LLM mode."

        # Step 4: Evaluate answer relevance and compute confidence score
        eval_result = evaluate_response(
            query=query, 
            answer=answer, 
            mode=mode, 
            retrieved_chunks=retrieved_chunks
        )
        confidence = eval_result.get("confidence", "Medium")

    except Exception as e:
        error_msg = str(e)
        mode = "general"
        answer = f"An error occurred while processing query: {error_msg}"
        confidence = "Low"

    # Step 5: Log execution stats (query, response time, mode, errors)
    elapsed_time = time.time() - start_time
    log_execution(
        query=query,
        mode=mode,
        response_time=elapsed_time,
        confidence=confidence,
        error=error_msg
    )

    # Step 6: Return structured JSON response matching exact requirements
    return {
        "mode": mode,
        "answer": answer.strip() if isinstance(answer, str) else str(answer),
        "confidence": confidence
    }
