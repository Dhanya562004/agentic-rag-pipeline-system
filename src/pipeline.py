"""
Pipeline Layer Module
Orchestrates Agent routing, Document Retrieval, Intelligent Fallback (threshold 0.75), LLM Response Generation, Evaluation, and Observability Logging.
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
    Executes RAG Document Retrieval returning top 3 clean chunks and generates context-grounded response.
    """
    retrieved_chunks = []
    top_score = 0.0
    context = ""

    if retriever:
        retrieved_chunks = retriever.hybrid_retrieve(query=query, top_k=3)
        if retrieved_chunks:
            top_score = max([chunk.get("similarity_score", 0.0) for chunk in retrieved_chunks])
            top_chunk_text = retrieved_chunks[0].get("text", "")
            print("TOP CHUNK:", top_chunk_text[:200])

            if top_score >= 0.75:
                context_parts = [chunk.get("text", "") for chunk in retrieved_chunks if chunk.get("text")]
                context = "\n\n".join(context_parts)
            else:
                print(f"DEBUG PIPELINE: Top similarity score {top_score:.3f} < 0.75 threshold. Skipping RAG context construction.")

    if llm_service and context:
        answer = llm_service.generate_response(query=query, context=context, max_tokens=200)
    elif context:
        answer = context[:300]
    else:
        answer = "Not found in context"

    return {
        "answer": answer.strip(),
        "retrieved_chunks": retrieved_chunks,
        "top_score": top_score
    }


def llm_pipeline(query: str, llm_service=None) -> dict:
    """
    Executes General Knowledge LLM Response Generation.
    """
    if llm_service:
        answer = llm_service.generate_general_response(query=query, max_tokens=200)
    else:
        answer = f"Response generated for query '{query}'."

    return {
        "answer": answer.strip()
    }


def run_pipeline(query: str, retriever=None, llm_service=None) -> dict:
    """
    Executes complete Agentic RAG Pipeline with 0.75 similarity threshold fallback.
    """
    start_time = time.time()
    error_msg = None

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
        initial_route = router(query_str)

        if initial_route == "rag":
            rag_res = rag_pipeline(query_str, retriever=retriever, llm_service=llm_service)
            answer = rag_res["answer"]
            retrieved_chunks = rag_res["retrieved_chunks"]
            top_score = rag_res["top_score"]

            eval_res = confidence_evaluator(
                query=query_str,
                answer=answer,
                mode="rag",
                retrieved_chunks=retrieved_chunks,
                top_score=top_score
            )

            # Fallback triggered if similarity < 0.75, no chunks, or 'not found in context'
            is_low_confidence = eval_res["confidence"] == "Low" or top_score < 0.75
            is_no_doc = "not found in context" in answer.lower() or "no relevant document" in answer.lower() or len(retrieved_chunks) == 0

            if is_low_confidence or is_no_doc:
                print(f"Notice: RAG Confidence low (Score: {top_score:.2f} < 0.75). Triggering Fallback to General LLM...")
                llm_res = llm_pipeline(query_str, llm_service=llm_service)
                answer = llm_res["answer"]
                mode = "RAG -> General LLM (Fallback)"

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
        answer = f"I am ready to help with your query: '{query_str}'."
        confidence = "Medium"
        score = 0.50

    print("Final Mode Decision:", mode)
    print("FINAL ANSWER:", answer)
    elapsed_time = time.time() - start_time
    log_execution(
        query=query_str,
        mode=mode,
        response_time=elapsed_time,
        confidence=confidence,
        score=score,
        error=error_msg
    )

    return {
        "mode": mode,
        "answer": answer,
        "confidence": confidence,
        "score": score,
        "retrieved_chunks": retrieved_chunks
    }
