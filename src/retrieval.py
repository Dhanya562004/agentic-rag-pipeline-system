"""
Retrieval Module for RAG Pipeline
Handles embedding generation and vector similarity search using sentence-transformers
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle
import os
import sys
import traceback
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class VectorRetriever:
    """
    Handles embedding generation and retrieval using sentence-transformers
    """

    def __init__(self, model_name: str = "togethercomputer/m2-bert-80M-8k-retrieval"):
        """
        Initialize the retriever with specified embedding model

        Args:
            model_name: Name of the sentence-transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        # Always use the data directory for embeddings
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.embeddings_file = os.path.join(self.data_dir, "embeddings.pkl")

        print(f"Initializing VectorRetriever with model: {model_name}")

    def load_model(self):
        """
        Load the sentence-transformer model
        """
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            try:
                # First try loading without trust_remote_code
                self.model = SentenceTransformer(self.model_name)
                print("Model loaded successfully!")
            except ValueError as e:
                # Retry with trust_remote_code=True if required
                if "trust_remote_code=True" in str(e):
                    print("Model requires trust_remote_code=True, retrying...")
                    try:
                        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
                        print("Model loaded successfully with trust_remote_code=True!")
                    except ImportError as ie:
                        # Try to extract missing package name
                        import re
                        error_msg = str(ie)
                        pkg_match = re.search(r"packages that were not found in your environment: ([a-zA-Z0-9_\-]+)", error_msg)
                        missing_pkg = pkg_match.group(1) if pkg_match else "unknown"
                        print(f"ERROR: Missing dependency: {missing_pkg}")
                        print(f"Install it with: pip install {missing_pkg}")
                        raise ImportError(f"Missing required dependency for {self.model_name}: {missing_pkg}")
                    except Exception as e2:
                        print(f"Error loading model with trust_remote_code=True: {e2}")
                        raise
                else:
                    raise e              

    def load_chunks(self, filename: str = "processed_data.json"):
        """
        Load processed text chunks

        Args:
            filename: Path to the processed data JSON file
        """
        # Always load from the data directory
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} chunks from {file_path}")

            # Validate chunk structure and essential metadata
            if self.chunks:
                if 'text' not in self.chunks[0]:
                    raise ValueError("Chunks missing 'text' field. Please check your processed data.")
                # Basic check for essential metadata used in retrieval/hybrid retrieval
                if not all(key in self.chunks[0] for key in ['title', 'category', 'section']):
                     print("Warning: Essential metadata fields (title, category, section) might be missing in processed_data.json. Retrieval might be affected.", file=sys.stderr)
                if 'date_retrieved' not in self.chunks[0]:
                    print("Warning: 'date_retrieved' metadata field is missing. 'boost_recent' functionality will not work.", file=sys.stderr)

        except FileNotFoundError:
            print(f"File {filename} not found. Please run data processing first.")
            self.chunks = []
        except Exception as e:
            print(f"Error loading chunks: {str(e)}")
            self.chunks = []

    def generate_embeddings(self, force_regenerate: bool = False):
        """
        Generate embeddings for all chunks

        Args:
            force_regenerate: If True, regenerate embeddings even if they exist
        """
        try:
            if not force_regenerate and os.path.exists(self.embeddings_file):
                print("Loading existing embeddings...")
                try:
                    with open(self.embeddings_file, 'rb') as f:
                        self.embeddings = pickle.load(f)
                    print(f"Loaded embeddings with shape: {self.embeddings.shape} from {self.embeddings_file}")
                    # Verify that the number of embeddings matches the number of chunks
                    if len(self.chunks) != self.embeddings.shape[0]:
                        print(f"Warning: Number of chunks ({len(self.chunks)}) doesn't match number of embeddings ({self.embeddings.shape[0]})")
                        print("Regenerating embeddings to ensure consistency...")
                        # Continue to regeneration code below
                    else:
                        return
                except Exception as e:
                    print(f"Error loading embeddings: {e}. Regenerating...")

            if not self.chunks:
                print("No chunks loaded. Cannot generate embeddings.")
                return

            self.load_model()

            print("Generating embeddings for all chunks...")
            texts = [chunk['text'] for chunk in self.chunks]

            # Generate embeddings in batches for efficiency
            batch_size = 32
            embeddings_list = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
                batch_embeddings = self.model.encode(batch, show_progress_bar=True)
                embeddings_list.append(batch_embeddings)

            # Combine all embeddings
            self.embeddings = np.vstack(embeddings_list)

            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)

            print(f"Generated and saved embeddings with shape: {self.embeddings.shape} to {self.embeddings_file}")
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            import traceback
            print(traceback.format_exc())
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query (pure vector search with optional pre-filter)

        Args:
            query: Search query
            top_k: Number of top results to return
            category_filter: Optional category to pre-filter by (e.g., "Climate", "Land")

        Returns:
            List of relevant chunks with similarity scores
        """
        try:
            if not self.chunks or self.embeddings is None:
                print("No chunks or embeddings available. Please load data and generate embeddings first.")
                return []

            self.load_model()
            query_embedding = self.model.encode([query])

            similarities = cosine_similarity(query_embedding, self.embeddings)[0]

            # Create list of (index, similarity, chunk) tuples, applying category filter early
            candidate_results = []
            for i, similarity in enumerate(similarities):
                chunk = self.chunks[i].copy() # Create a copy to add similarity_score without modifying original
                
                # Apply category filter if specified
                if category_filter:
                    chunk_category = chunk.get('category', '').lower()
                    if chunk_category != category_filter.lower():
                        continue # Skip this chunk if it doesn't match the filter
                
                chunk['similarity_score'] = float(similarity)
                candidate_results.append((i, similarity, chunk))

            # Sort by similarity score (descending)
            candidate_results.sort(key=lambda x: x[1], reverse=True)

            top_results = []
            for i, (idx, sim_score, chunk) in enumerate(candidate_results[:top_k]):
                result = {
                    'rank': i + 1,
                    'chunk_id': chunk.get('chunk_id', idx),
                    'text': chunk['text'],
                    'title': chunk.get('title', 'Unknown'),
                    'category': chunk.get('category', 'Unknown'),
                    'section': chunk.get('section', 'Unknown'),
                    'source_url': chunk.get('source_url', ''),
                    'similarity_score': float(sim_score),
                    'word_count': chunk.get('word_count', 0),
                    'date_retrieved': chunk.get('date_retrieved', 'N/A')
                }
                top_results.append(result)
                
            return top_results
        
        except Exception as e:
            print(f"Error in retrieve method: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def hybrid_retrieve(self, query: str, top_k: int = 5,
                        category_filter: Optional[str] = None,
                        boost_recent: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval combining vector similarity with metadata filtering and boosting.

        Args:
            query: Search query
            top_k: Number of top results to return
            category_filter: Optional category to filter by (e.g., "Climate", "Land")
            boost_recent: Whether to boost more recent content based on 'date_retrieved' metadata.

        Returns:
            List of relevant chunks with enhanced scoring
        """
        try:
            results = self.retrieve(query, top_k * 5, category_filter)

            if not results:
                print("No candidates found for hybrid retrieval.")
                return []

            # Apply hybrid scoring
            for result in results:
                hybrid_score = result['similarity_score']

                # Boost based on word count (longer chunks might be more informative)
                if result['word_count'] > 300:
                    hybrid_score *= 1.1

                # Boost based on title/section relevance
                query_lower = query.lower()
                title_lower = result.get('title', '').lower()
                section_lower = result.get('section', '').lower()

                # Prioritize exact phrase match, then word-by-word match
                if query_lower in title_lower:
                    hybrid_score *= 1.3 # Stronger boost for direct title match
                elif any(word in title_lower for word in query_lower.split()):
                    hybrid_score *= 1.2 # Word match

                if query_lower in section_lower:
                    hybrid_score *= 1.25 # Stronger boost for direct section match
                elif any(word in section_lower for word in query_lower.split()):
                    hybrid_score *= 1.15 # Word match

                # Date boosting logic
                if boost_recent and result.get('date_retrieved') and result['date_retrieved'] != 'N/A':
                    try:
                        # Try multiple date formats to be more flexible
                        date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
                        chunk_date = None
                        
                        for date_format in date_formats:
                            try:
                                chunk_date = datetime.strptime(result['date_retrieved'], date_format)
                                break
                            except ValueError:
                                continue
                        
                        if chunk_date:
                            current_time = datetime.now()
                            days_ago = (current_time - chunk_date).days
                            
                            # Apply a boost for more recent content
                            # Linear decay: 0% boost for 2 years ago, 15% boost for today
                            if days_ago < 365 * 2:
                                recency_boost_factor = 1 + (1 - (days_ago / (365 * 2))) * 0.15
                                hybrid_score *= recency_boost_factor
                    except Exception as e:
                        print(f"Warning: Error in date boosting for chunk ID {result.get('chunk_id')}: {str(e)}")

                result['hybrid_score'] = float(hybrid_score)

            results.sort(key=lambda x: x['hybrid_score'], reverse=True)

            return results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid_retrieve method: {str(e)}")
            print(traceback.format_exc())
            return []

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding space

        Returns:
            Dictionary with embedding statistics
        """
        if self.embeddings is None:
            return {"error": "No embeddings available"}

        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1],
            "model_name": self.model_name,
            "embedding_file_exists": os.path.exists(self.embeddings_file)
        }

def create_embeddings():
    """Utility function to create embeddings from processed data"""
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings(force_regenerate=True) # Force regenerate to ensure fresh embeddings
    
    stats = retriever.get_embedding_statistics()
    print("\nEmbedding Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")

def test_retrieval():
    """Test the retrieval functionality, including hybrid retrieval and top-K tuning."""
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings()

    # --- Test Vector Retrieval (top-K tuning demonstration) ---
    print("\n" + "="*70)
    print("TESTING VECTOR RETRIEVAL: TOP-K TUNING & BASIC FUNCTIONALITY")
    print("="*70)

    test_queries_vector = [
        "What is the typical rainfall in the lowlands of France?",
        "Describe the soil characteristics found in the central Paris Basin.",
        "Which two major French river systems are separated by the Jurassic limestone Plateau de Langres?"
    ]

    for query in test_queries_vector:
        print(f"\nQuery: '{query}'")
        print("-" * 30)

        # Compare top_k = 3
        print("--- Top-K = 3 (Pure Vector Search) ---")
        results_k3 = retriever.retrieve(query, top_k=3)
        if not results_k3: print("No results found.")
        for i, result in enumerate(results_k3, 1):
            print(f"{i}. [Sim: {result['similarity_score']:.3f}] Title: {result['title']} | Category: {result['category']} | Section: {result['section']}")
            print(f"    Text: {result['text'][:150]}...") # Show a bit more text
            print(f"    Source: {result['source_url']} | Date: {result['date_retrieved']}")

    # --- Test Hybrid Retrieval Demonstration ---
    print("\n" + "="*70)
    print("TESTING HYBRID RETRIEVAL: METADATA FILTERING & BOOSTING")
    print("="*70)

    # Example test cases for hybrid retrieval
    # Adjust category_filter values to match your actual categories in processed_data.json
    hybrid_test_cases = [
        # Case 1: Strong category filter, no recency boost
        {"query": "climate of France's lowlands", "category_filter": "climate", "boost_recent": False, "description": "Category filter for 'Climate'"},
        # Case 2: Query for a specific geological feature, using category filter
        {"query": "formation of Hercynian massifs", "category_filter": "the-hercynian-massifs", "boost_recent": False, "description": "Category filter for 'The Hercynian Massifs'"},
        # Case 3: More general query, no category filter, no recency boost (relying on hybrid scoring)
        {"query": "characteristics of French soils", "category_filter": None, "boost_recent": False, "description": "General query, no explicit filter (hybrid scoring only)"},
        # Case 4: Test recency boost (requires 'date_retrieved' in your data)
        # Assuming you have recent chunks, this should boost them.
        # {"query": "recent details on french climate", "category_filter": "climate", "boost_recent": True, "description": "Category filter + recency boost"},
        # Case 5: A general query, no category filter, with recency boost
        # {"query": "latest info on french landforms", "category_filter": None, "boost_recent": True, "description": "General query + recency boost"}
    ]

    for test_case in hybrid_test_cases:
        query = test_case['query']
        category_filter = test_case['category_filter']
        boost_recent = test_case['boost_recent']
        description = test_case['description']

        print(f"\nQuery: '{query}' ({description})")
        print(f"  Parameters: Category Filter: {category_filter}, Boost Recent: {boost_recent}")
        print("-" * 30)

        # Retrieve with hybrid method (using top_k=3 for demonstration)
        hybrid_results = retriever.hybrid_retrieve(query, top_k=3,
                                                   category_filter=category_filter,
                                                   boost_recent=boost_recent)

        if not hybrid_results:
            print("No results found for this hybrid query.")
            continue

        for i, result in enumerate(hybrid_results, 1):
            score_to_print = result.get('hybrid_score', result.get('similarity_score', 0.0))
            score_label = "Hybrid" if 'hybrid_score' in result else "Sim" # Indicate if hybrid score was applied

            print(f"{i}. [{score_label}: {score_to_print:.3f}] Title: {result['title']} | Category: {result['category']} | Section: {result['section']}")
            print(f"    Text: {result['text'][:150]}...")
            print(f"    Source: {result['source_url']} | Date: {result['date_retrieved']}")
            if 'similarity_score' in result and 'hybrid_score' in result:
                 print(f"    (Base Sim: {result['similarity_score']:.3f})") # Show base sim for comparison
            print()

    print("\n" + "="*70)
    print("ALL RETRIEVAL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    # 1. Create/regenerate embeddings (run this once after data processing)
    # Set force_regenerate=True to ensure fresh embeddings if you update data.
    create_embeddings()

    # 2. Run retrieval tests to see performance and gather data for justification
    test_retrieval()