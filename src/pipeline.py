from retrieval import VectorRetriever
from llm_service import LLMService

def run_pipeline(query: str, retriever: VectorRetriever, llm_service: LLMService) -> dict:
    """
    RAG Pipeline Decision Logic:
    1. If no document/chunks stored -> Mode: "🔵 LLM"
    2. Retrieve top 3 chunks & similarity score:
       - score > 0.5 -> Mode: "🟢 RAG" (answer from context)
       - score <= 0.5 -> Mode: "🟡 RAG → LLM (Fallback)" (general AI response)
    """
    if not query or not query.strip():
        return {
            "mode": "🔵 LLM",
            "score": 0.0,
            "answer": "Please enter a valid question.",
            "retrieved_chunks": []
        }

    # If no document is uploaded or indexed
    if not retriever or not retriever.chunks:
        answer = llm_service.generate_response(prompt=query, context=None)
        return {
            "mode": "🔵 LLM",
            "score": 0.0,
            "answer": answer,
            "retrieved_chunks": []
        }

    # Cosine similarity search for top 3 chunks
    retrieved_chunks, max_score = retriever.search(query, top_k=3)

    if max_score > 0.5:
        # Score > 0.5 -> RAG Mode
        context_text = "\n\n---\n\n".join([chunk["text"] for chunk in retrieved_chunks])
        answer = llm_service.generate_response(prompt=query, context=context_text)
        return {
            "mode": "🟢 RAG",
            "score": max_score,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }
    else:
        # Score <= 0.5 -> Fallback to General LLM Response
        answer = llm_service.generate_response(prompt=query, context=None)
        return {
            "mode": "🟡 RAG → LLM (Fallback)",
            "score": max_score,
            "answer": answer,
            "retrieved_chunks": retrieved_chunks
        }
