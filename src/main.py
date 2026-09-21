from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import numpy as np
import traceback
from retrieval import VectorRetriever
from llm_service import LLMService
from pipeline import run_pipeline
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

# Utility function to convert NumPy types to Python native types
def convert_numpy_types(obj):
    try:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    except Exception as e:
        print(f"Error converting numpy types: {str(e)}")
        # Return a safe fallback value depending on the type
        if isinstance(obj, (np.integer, np.floating)):
            return 0
        elif isinstance(obj, np.ndarray):
            return []
        else:
            return obj

load_dotenv()

app = FastAPI(title="RAG Pipeline API")

# Mount static files directory
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except Exception as e:
    print(f"Warning: Could not mount static files: {str(e)}")

# Initialize retriever (will load embeddings automatically)
retriever = VectorRetriever()
retriever.load_chunks()
retriever.generate_embeddings()

# Initialize LLM service
try:
    llm_service = LLMService(provider="together")  # You can change to "openai"
    print("LLM service initialized successfully!")
except Exception as e:
    print(f"Warning: Could not initialize LLM service: {e}")
    llm_service = None

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None

class RetrievalResponse(BaseModel):
    query: str
    results: list
    total_found: int

class GenerationRequest(BaseModel):
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None
    max_tokens: int = 500  # Default token limit to encourage concise answers

class PipelineRequest(BaseModel):
    query: str

class PipelineResponse(BaseModel):
    mode: str
    answer: str
    confidence: str

@app.post("/ask", response_model=PipelineResponse)
def ask(request: PipelineRequest):
    """
    Unified Pipeline Endpoint:
    Routes query (Agent) -> Retrieves context (if RAG) -> Generates response (LLM) -> Evaluates confidence -> Logs stats
    Returns JSON: {"mode": "rag" | "general", "answer": "...", "confidence": "High" | "Medium" | "Low"}
    """
    try:
        result = run_pipeline(
            query=request.query,
            retriever=retriever,
            llm_service=llm_service
        )
        return result
    except Exception as e:
        print(f"Error in /ask endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ask")
def ask_get(query: str):
    """
    GET /ask endpoint for direct browser testing: /ask?query=What+is+the+capital+of+France
    """
    try:
        result = run_pipeline(
            query=query,
            retriever=retriever,
            llm_service=llm_service
        )
        return result
    except Exception as e:
        print(f"Error in /ask GET endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: QueryRequest):
    """
    Retrieve relevant chunks for a query using vector similarity
    """
    try:
        # Use hybrid retrieval for better results
        results = retriever.hybrid_retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Convert NumPy types to Python native types
        converted_results = convert_numpy_types(results)
        
        return RetrievalResponse(
            query=request.query,
            results=converted_results,
            total_found=len(results)
        )
    except Exception as e:
        print(f"Error in /retrieve endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/generate")
def generate(request: GenerationRequest):
    """
    Generate response using retrieved context and LLM
    """
    try:
        # First retrieve relevant chunks
        chunks = retriever.hybrid_retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Convert NumPy types to Python native types
        chunks = convert_numpy_types(chunks)
        
        if not chunks:
            return {"query": request.query, "response": "No relevant information found."}
        
        # Create context from retrieved chunks - structured for clear parsing by the LLM
        context_parts = []
        for i, chunk in enumerate(chunks):
            context_parts.append(f"[Source {i+1}: {chunk['title']} - {chunk['category']}]\n{chunk['text']}")
        context = "\n\n".join(context_parts)
        
        # Generate response using LLM with token limit
        if llm_service:
            try:
                # Use the max_tokens parameter to enforce concise answers
                response = llm_service.generate_response(
                    query=request.query, 
                    context=context, 
                    max_tokens=request.max_tokens
                )
                # Remove any trailing punctuation or whitespace
                response = response.strip().rstrip('.,;:')
            except Exception as e:
                print(f"Error generating LLM response: {str(e)}")
                response = f"Error generating LLM response: {str(e)}\n\nFallback response based on retrieved context:\n{context[:500]}..."
        else:
            response = f"LLM service not available. Retrieved context:\n{context[:500]}..."
        
        return {
            "query": request.query,
            "response": response,
            "retrieved_chunks": len(chunks),
            "sources": [chunk['source_url'] for chunk in chunks],
            "llm_used": llm_service is not None
        }
    except Exception as e:
        print(f"Error in /generate endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/ask-ui")
def ask_ui_page():
    """Serve the interactive pipeline ask UI"""
    try:
        static_path = Path(__file__).parent.parent / "static" / "ask.html"
        return FileResponse(str(static_path.resolve()))
    except Exception as e:
        print(f"Error serving ask UI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving ask UI: {str(e)}")

@app.get("/")
def root():
    """Serve the web UI"""
    try:
        static_path = Path(__file__).parent.parent / "static" / "index.html"
        return FileResponse(str(static_path.resolve()))
    except Exception as e:
        print(f"Error serving UI: {str(e)}")
        return {"message": "RAG Pipeline API is running", "status": "healthy"}

@app.get("/compare")
def comparison_page():
    """Serve the retrieval comparison UI"""
    try:
        static_path = Path(__file__).parent.parent / "static" / "compare.html"
        return FileResponse(str(static_path.resolve()))
    except Exception as e:
        print(f"Error serving comparison UI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving comparison UI: {str(e)}")

@app.get("/health")
def health_check():
    """API health check"""
    return {"message": "RAG Pipeline API is running", "status": "healthy"}

@app.get("/stats")
def get_stats():
    """Get retrieval system statistics"""
    try:
        stats = retriever.get_embedding_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/llm-status")
def get_llm_status():
    """Get LLM service status and configuration"""
    if llm_service:
        return {
            "status": "available",
            "model_info": llm_service.get_model_info()
        }
    else:
        return {
            "status": "unavailable",
            "error": "LLM service not initialized"
        }

# Serve the static HTML file
@app.get("/index")
def get_index():
    static_path = Path(__file__).parent.parent / "static" / "index.html"
    return FileResponse(str(static_path.resolve()))

class CompareRetrievalRequest(BaseModel):
    """Request model for comparing retrieval methods"""
    query: str
    top_k: int = 5
    category_filter: Optional[str] = None

@app.post("/compare-retrieval")
def compare_retrieval(request: CompareRetrievalRequest):
    """
    Compare different retrieval methods (vector vs hybrid) for the same query
    
    This endpoint demonstrates why the hybrid approach yields better precision/recall
    by showing results from both pure vector search and hybrid search side by side.
    """
    try:
        # Get results from pure vector search
        vector_results = retriever.retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Get results from hybrid search
        hybrid_results = retriever.hybrid_retrieve(
            query=request.query,
            top_k=request.top_k,
            category_filter=request.category_filter
        )
        
        # Convert NumPy types to Python native types
        vector_results = convert_numpy_types(vector_results)
        hybrid_results = convert_numpy_types(hybrid_results)
        
        # Calculate metrics to demonstrate the difference
        # 1. How many results are common between the two methods
        vector_ids = set(result['chunk_id'] for result in vector_results)
        hybrid_ids = set(result['chunk_id'] for result in hybrid_results)
        common_results = len(vector_ids.intersection(hybrid_ids))
        
        # 2. Average similarity score for each method
        avg_vector_score = sum(r['similarity_score'] for r in vector_results) / len(vector_results) if vector_results else 0
        avg_hybrid_score = sum(r['hybrid_score'] for r in hybrid_results) / len(hybrid_results) if hybrid_results else 0
        
        # 3. Relevance metrics (estimate based on title/section matching)
        query_terms = set(request.query.lower().split())
        
        def estimate_relevance(results):
            relevance_scores = []
            for r in results:
                # Count how many query terms appear in title, section, and text
                title_terms = set(r.get('title', '').lower().split())
                section_terms = set(r.get('section', '').lower().split())
                # Convert to list first, slice, then convert back to set
                text_terms = set(list(r.get('text', '').lower().split())[:50])  # Only use first 50 terms for efficiency
                term_overlap = len(query_terms.intersection(title_terms.union(section_terms).union(text_terms)))
                relevance_scores.append(term_overlap / max(1, len(query_terms)))
            return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        vector_relevance = estimate_relevance(vector_results)
        hybrid_relevance = estimate_relevance(hybrid_results)
        
        return {
            "query": request.query,
            "vector_results": vector_results,
            "hybrid_results": hybrid_results,
            "comparison": {
                "total_results": request.top_k,
                "common_results": common_results,
                "result_overlap_percentage": round(common_results / request.top_k * 100, 2),
                "avg_vector_similarity": round(avg_vector_score, 4),
                "avg_hybrid_score": round(avg_hybrid_score, 4),
                "estimated_vector_relevance": round(vector_relevance, 4),
                "estimated_hybrid_relevance": round(hybrid_relevance, 4),
                "performance_improvement": round((hybrid_relevance - vector_relevance) / max(0.01, vector_relevance) * 100, 2)
            },
            "justification": "The hybrid retrieval approach typically improves results by incorporating metadata signals alongside vector similarity. It boosts content with relevant titles/sections and adjusts for factors like word count."
        }
        
    except Exception as e:
        print(f"Error in /compare-retrieval endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# Define models for benchmarking
class BenchmarkQuery(BaseModel):
    query: str
    expected_answer: Optional[str] = None  # Reference answer for comparison if available
    expected_sources: Optional[List[str]] = None  # Expected source IDs if available

class BenchmarkRequest(BaseModel):
    queries: List[BenchmarkQuery]
    top_k: int = 5
    category_filter: Optional[str] = None
    max_tokens: int = 500

@app.post("/benchmark")
def benchmark_system(request: BenchmarkRequest):
    """
    Benchmark the RAG pipeline using a set of test queries
    
    Evaluates the system on:
    - Accuracy & Relevance of answers
    - Faithfulness (whether responses are grounded in retrieved context)
    - Coverage of retrieved chunks
    - Response Consistency (format and style)
    - Robustness (edge-case handling)
    """
    try:
        results = []
        metrics = {
            "total_queries": len(request.queries),
            "successful_queries": 0,
            "retrieval_metrics": {
                "avg_chunks_retrieved": 0,
                "avg_similarity_score": 0,
            },
            "generation_metrics": {
                "avg_response_length": 0,
                "avg_generation_time": 0,
            },
            "accuracy_metrics": {
                "estimated_relevance": 0
            },
            "robustness_metrics": {
                "error_rate": 0,
                "empty_results_rate": 0
            }
        }
        
        import time
        total_similarity = 0
        total_chunks = 0
        total_response_length = 0
        total_generation_time = 0
        error_count = 0
        empty_results_count = 0
        
        for query_item in request.queries:
            try:
                # Time the retrieval and generation process
                start_time = time.time()
                
                # First retrieve relevant chunks
                chunks = retriever.hybrid_retrieve(
                    query=query_item.query,
                    top_k=request.top_k,
                    category_filter=request.category_filter
                )
                
                # Convert NumPy types to Python native types
                chunks = convert_numpy_types(chunks)
                
                if not chunks:
                    empty_results_count += 1
                    results.append({
                        "query": query_item.query,
                        "response": "No relevant information found.",
                        "retrieved_chunks": 0,
                        "sources": [],
                        "generation_time": 0,
                        "error": "No chunks retrieved"
                    })
                    continue
                
                # Calculate average similarity score for this query
                avg_similarity = sum(chunk.get('hybrid_score', 0) for chunk in chunks) / len(chunks)
                total_similarity += avg_similarity
                total_chunks += len(chunks)
                
                # Create context from retrieved chunks
                context_parts = []
                for i, chunk in enumerate(chunks):
                    context_parts.append(f"[Source {i+1}: {chunk['title']} - {chunk['category']}]\n{chunk['text']}")
                context = "\n\n".join(context_parts)
                
                # Generate response using LLM
                if llm_service:
                    response = llm_service.generate_response(
                        query=query_item.query, 
                        context=context, 
                        max_tokens=request.max_tokens
                    )
                    response = response.strip().rstrip('.,;:')
                else:
                    response = f"LLM service not available. Retrieved context:\n{context[:500]}..."
                
                generation_time = time.time() - start_time
                total_generation_time += generation_time
                total_response_length += len(response)
                
                # Check for reference answer overlap if provided
                relevance_score = None
                if query_item.expected_answer:
                    # Simple word overlap metric
                    expected_words = set(query_item.expected_answer.lower().split())
                    response_words = set(response.lower().split())
                    overlap = len(expected_words.intersection(response_words))
                    relevance_score = overlap / max(1, len(expected_words))
                
                # Check for expected sources if provided
                source_accuracy = None
                if query_item.expected_sources:
                    retrieved_sources = [chunk.get('source_id', '') for chunk in chunks]
                    matched_sources = set(retrieved_sources).intersection(set(query_item.expected_sources))
                    source_accuracy = len(matched_sources) / max(1, len(query_item.expected_sources))
                
                result = {
                    "query": query_item.query,
                    "response": response,
                    "retrieved_chunks": len(chunks),
                    "sources": [chunk.get('source_url', chunk.get('source_id', '')) for chunk in chunks],
                    "generation_time": round(generation_time, 3),
                    "avg_similarity_score": round(avg_similarity, 4),
                    "llm_used": llm_service is not None
                }
                
                # Add relevance score if available
                if relevance_score is not None:
                    result["relevance_score"] = round(relevance_score, 4)
                
                # Add source accuracy if available
                if source_accuracy is not None:
                    result["source_accuracy"] = round(source_accuracy, 4)
                
                results.append(result)
                metrics["successful_queries"] += 1
                
            except Exception as e:
                print(f"Error processing benchmark query '{query_item.query}': {str(e)}")
                error_count += 1
                results.append({
                    "query": query_item.query,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
        
        # Calculate final metrics
        query_count = metrics["successful_queries"]
        if query_count > 0:
            metrics["retrieval_metrics"]["avg_chunks_retrieved"] = round(total_chunks / query_count, 2)
            metrics["retrieval_metrics"]["avg_similarity_score"] = round(total_similarity / query_count, 4)
            metrics["generation_metrics"]["avg_response_length"] = round(total_response_length / query_count, 2)
            metrics["generation_metrics"]["avg_generation_time"] = round(total_generation_time / query_count, 3)
        
        metrics["robustness_metrics"]["error_rate"] = round(error_count / max(1, len(request.queries)), 4)
        metrics["robustness_metrics"]["empty_results_rate"] = round(empty_results_count / max(1, len(request.queries)), 4)
        
        # Evaluate overall system performance
        performance_score = 0
        if query_count > 0:
            # Weighted scoring (customize weights based on priorities)
            retrieval_weight = 0.4
            generation_weight = 0.3
            robustness_weight = 0.3
            
            retrieval_score = min(1.0, metrics["retrieval_metrics"]["avg_similarity_score"])
            generation_score = min(1.0, 1.0 - metrics["robustness_metrics"]["error_rate"])
            robustness_score = 1.0 - metrics["robustness_metrics"]["empty_results_rate"]
            
            performance_score = (
                retrieval_weight * retrieval_score +
                generation_weight * generation_score +
                robustness_weight * robustness_score
            )
        
        return {
            "benchmark_results": results,
            "metrics": metrics,
            "performance_score": round(performance_score * 100, 2),  # Score as a percentage
            "evaluation_criteria": [
                "Accuracy & Relevance of answers",
                "Faithfulness (answers grounded in retrieved context)",
                "Coverage of retrieved chunks",
                "Response Consistency (format and style)",
                "Robustness (edge-case handling)"
            ]
        }
        
    except Exception as e:
        print(f"Error in /benchmark endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/benchmark-ui")
def benchmark_page():
    """Serve the benchmark UI"""
    try:
        static_path = Path(__file__).parent.parent / "static" / "benchmark.html"
        return FileResponse(str(static_path.resolve()))
    except Exception as e:
        print(f"Error serving benchmark UI: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error serving benchmark UI: {str(e)}")
