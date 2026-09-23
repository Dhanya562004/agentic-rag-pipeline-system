"""
Pipeline Layer Module
Orchestrates Agent routing, Document Retrieval, Intelligent Fallback, LLM Response Generation, Evaluation, and Observability Logging.
"""

import time
import os
import sys

# Ensure src directory is in sys.path for clean module resolution
src_dir = os.path.dirname(__file__)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from agent import router, route_query
from evaluator import confidence_evaluator, evaluate_response
from logger import log_execution

def rag_pipeline(query: str, retriever=None, llm_service=None) -> dict:
    """
    Executes RAG Document Retrieval and generates context-grounded response.
    
    Args:
        query: User input prompt
        retriever: Instance of VectorRetriever
        llm_service: Instance of LLMService
        
    Returns:
        dict: {"answer": str, "retrieved_chunks": list, "top_score": float}
    """
    retrieved_chunks = []
    top_score = 0.0
    context = ""
    
    if retriever:
        retrieved_chunks = retriever.hybrid_retrieve(query=query, top_k=5)
        if retrieved_chunks:
            top_score = max([chunk.get("similarity_score", 0.0) for chunk in retrieved_chunks])
            context_parts = [
                f"[Source {i+1}: {chunk.get('title', 'Doc')} - {chunk.get('category', 'General')}]\n{chunk.get('text', '')}"
                for i, chunk in enumerate(retrieved_chunks)
            ]
            context = "\n\n".join(context_parts)
            
    if llm_service and context:
        answer = llm_service.generate_response(query=query, context=context, max_tokens=500)
    elif context:
        answer = f"Retrieved Context:\n{context[:400]}..."
    else:
        answer = "No relevant document found in knowledge base."
        
    return {
        "answer": answer.strip(),
        "retrieved_chunks": retrieved_chunks,
        "top_score": top_score
    }

def llm_pipeline(query: str, llm_service=None) -> dict:
    """
    Executes General Knowledge LLM Response Generation.
    
    Args:
        query: User input prompt
        llm_service: Instance of LLMService
        
    Returns:
        dict: {"answer": str}
    """
    if llm_service:
        answer = llm_service.generate_general_response(query=query, max_tokens=500)
    else:
        answer = f"General LLM Knowledge: Response generated for query '{query}'."
        
    return {
        "answer": answer.strip()
    }

def run_pipeline(query: str, retriever=None, llm_service=None) -> dict:
    """
    Executes complete Agentic RAG Pipeline with Intelligent Fallback Logic.
    
    Flow:
    1. Route query intent using router() -> 'rag' or 'general'.
    2. If 'rag', execute rag_pipeline().
    3. Evaluate retrieval confidence score (threshold < 0.5).
    4. If RAG returns no chunks, low score (< 0.5), or 'No relevant document found':
       AUTOMATICALLY FALLBACK to llm_pipeline()!
    5. Evaluate confidence rating and log execution.
    6. Return structured dictionary.
    
    Args:
        query: User input prompt
        retriever: Instance of VectorRetriever
        llm_service: Instance of LLMService
        
    Returns:
        dict: {
            "mode": "RAG" | "General LLM" | "RAG -> General LLM (Fallback)",
            "answer": str,
            "confidence": "High" | "Medium" | "Low",
            "score": float,
            "retrieved_chunks": list
        }
    """
    start_time = time.time()
    error_msg = None
    
    # Handle empty input
    if not query or not query.strip():
        return {
            "mode": "General LLM",
            "answer": "Please enter a valid non-empty question.",
            "confidence": "Low",
            "score": 0.0,
            "retrieved_chunks": []
        }

    query_str = query.strip()
    retrieved_chunks = []
    top_score = 0.0

    try:
        # Step 1: Agent routes query intent
        initial_route = router(query_str)
        
        if initial_route == "rag":
            # Step 2: RAG Pipeline Execution
            rag_res = rag_pipeline(query_str, retriever=retriever, llm_service=llm_service)
            answer = rag_res["answer"]
            retrieved_chunks = rag_res["retrieved_chunks"]
            top_score = rag_res["top_score"]
            
            # Step 3: Evaluate initial RAG confidence
            eval_res = confidence_evaluator(
                query=query_str, 
                answer=answer, 
                mode="rag", 
                retrieved_chunks=retrieved_chunks, 
                top_score=top_score
            )
            
            # Step 4: Intelligent Fallback Trigger Check
            # Fallback triggered if: 0 chunks OR top similarity score < 0.5 OR low confidence / no relevant doc
            is_low_confidence = eval_res["confidence"] == "Low" or top_score < 0.5
            is_no_doc = "no relevant document" in answer.lower() or len(retrieved_chunks) == 0
            
            if is_low_confidence or is_no_doc:
                print(f"Notice: RAG Confidence low (Score: {top_score:.2f}). Triggering Fallback to General LLM...")
                llm_res = llm_pipeline(query_str, llm_service=llm_service)
                answer = llm_res["answer"]
                mode = "RAG -> General LLM (Fallback)"
                
                # Re-evaluate confidence for fallback mode
                eval_res = confidence_evaluator(
                    query=query_str, 
                    answer=answer, 
                    mode=mode, 
                    retrieved_chunks=retrieved_chunks, 
                    top_score=top_score
                )
            else:
                mode = "RAG"
        else:
            # Step 5: General LLM Pipeline Execution
            mode = "General LLM"
            llm_res = llm_pipeline(query_str, llm_service=llm_service)
            answer = llm_res["answer"]
            eval_res = confidence_evaluator(
                query=query_str, 
                answer=answer, 
                mode=mode
            )

        confidence = eval_res.get("confidence", "Medium")
        score = eval_res.get("score", 0.0)

    except Exception as e:
        error_msg = str(e)
        print(f"Pipeline error: {error_msg}")
        mode = "General LLM (Fallback)"
        answer = f"I am ready to help with your query: '{query_str}'. System operated in general fallback mode."
        confidence = "Medium"
        score = 0.50

    # Step 6: Log execution metrics
    elapsed_time = time.time() - start_time
    log_execution(
        query=query_str,
        mode=mode,
        response_time=elapsed_time,
        confidence=confidence,
        score=score,
        error=error_msg
    )

    # Return structured result
    return {
        "mode": mode,
        "answer": answer,
        "confidence": confidence,
        "score": score,
        "retrieved_chunks": retrieved_chunks
    }

