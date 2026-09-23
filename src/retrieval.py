"""
Retrieval Module for RAG Pipeline
Handles embedding generation, text cleaning, chunk filtering, and vector similarity search using sentence-transformers
"""

import numpy as np
import json
import re
from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SentenceTransformer = None
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import pickle
import os
import sys
import traceback
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

NOISE_KEYWORDS = [
    "Quiz", "Feedback", "Print", "Ask the Chatbot", "Related Questions",
    "External Websites", "Table Of Contents", "Table of Contents", "Cite",
    "Citation", "Share", "Subscribe", "Encyclopaedia Britannica", "Written by",
    "Fact-checked by", "Copyright", "All rights reserved", "Article History",
    "Select a type", "Submit Feedback", "Which Country Is Larger", "Photo Gallery",
    "Children's Encyclopedia", "Student Encyclopedia", "Facts & Stats", "Images, Videos & Interactives"
]

def clean_text(text: str) -> str:
    """
    Clean text content by removing UI noise, quiz text, navigation blocks, duplicate lines, and uninformative headers.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned, high-quality text string
    """
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []
    seen_lines = set()

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue

        # Check if line contains any noise keywords
        if any(kw.lower() in line_str.lower() for kw in NOISE_KEYWORDS):
            continue

        # Ignore short uninformative navigation strings
        if len(line_str) < 5 and not line_str.endswith(('.', '!', '?')):
            continue

        # Deduplicate identical lines
        line_key = line_str.lower()
        if line_key in seen_lines:
            continue
        seen_lines.add(line_key)
        cleaned_lines.append(line_str)

    cleaned_text = " ".join(cleaned_lines)
    
    # Remove leftover inline noise phrases
    for kw in NOISE_KEYWORDS:
        pattern = re.compile(re.escape(kw) + r'[^.!?]*', re.IGNORECASE)
        cleaned_text = pattern.sub('', cleaned_text)

    # Normalize spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text


class VectorRetriever:
    """
    Handles embedding generation and retrieval using sentence-transformers with clean text filtering.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the retriever with specified embedding model
        """
        self.model_name = model_name
        self.model = None
        self.chunks = []
        self.embeddings = None
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.embeddings_file = os.path.join(self.data_dir, "embeddings.pkl")

        print(f"Initializing VectorRetriever with model: {model_name}")

    def load_model(self):
        """
        Load the sentence-transformer model with robust fallback handling.
        """
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            try:
                try:
                    self.model = SentenceTransformer(self.model_name, trust_remote_code=True)
                except Exception:
                    self.model = SentenceTransformer(self.model_name)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Notice: Failed loading {self.model_name}: {e}. Falling back to 'all-MiniLM-L6-v2'...")
                self.model_name = "all-MiniLM-L6-v2"
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                print("Fallback model 'all-MiniLM-L6-v2' loaded successfully!")

    def clean_text(self, text: str) -> str:
        """Instance method wrapper for module-level clean_text"""
        return clean_text(text)

    def load_chunks(self, filename: str = "processed_data.json"):
        """
        Load processed text chunks, clean text, and discard chunks with length < 50 characters or containing noise keywords.

        Args:
            filename: Path to the processed data JSON file
        """
        file_path = os.path.join(self.data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)

            filtered_chunks = []
            for chunk in raw_chunks:
                raw_text = chunk.get('text', '')
                
                # Filter out chunks containing noise keywords in raw text
                if any(kw.lower() in raw_text.lower() for kw in NOISE_KEYWORDS):
                    continue

                cleaned = clean_text(raw_text)
                
                # Check if cleaned text still contains any noise keyword or is < 50 characters
                if any(kw.lower() in cleaned.lower() for kw in NOISE_KEYWORDS):
                    continue

                # Discard chunks with length < 50 characters
                if len(cleaned) >= 50:
                    chunk_copy = chunk.copy()
                    chunk_copy['text'] = cleaned
                    chunk_copy['word_count'] = len(cleaned.split())
                    filtered_chunks.append(chunk_copy)

            self.chunks = filtered_chunks
            print(f"Loaded {len(self.chunks)} clean chunks (filtered from {len(raw_chunks)} raw chunks) from {file_path}")

        except FileNotFoundError:
            print(f"File {filename} not found. Please run data processing first.")
            self.chunks = []
        except Exception as e:
            print(f"Error loading chunks: {str(e)}")
            self.chunks = []

    def generate_embeddings(self, force_regenerate: bool = False):
        """
        Generate embeddings for all clean chunks
        """
        try:
            if not force_regenerate and os.path.exists(self.embeddings_file):
                print("Loading existing embeddings...")
                try:
                    with open(self.embeddings_file, 'rb') as f:
                        self.embeddings = pickle.load(f)
                    print(f"Loaded embeddings with shape: {self.embeddings.shape} from {self.embeddings_file}")
                    if len(self.chunks) != self.embeddings.shape[0]:
                        print(f"Warning: Number of clean chunks ({len(self.chunks)}) doesn't match embeddings count ({self.embeddings.shape[0]}). Regenerating...")
                        force_regenerate = True
                    else:
                        return
                except Exception as e:
                    print(f"Error loading embeddings: {e}. Regenerating...")
                    force_regenerate = True

            if not self.chunks:
                print("No clean chunks loaded. Cannot generate embeddings.")
                return

            self.load_model()
            print("Generating embeddings for clean chunks...")
            texts = [chunk['text'] for chunk in self.chunks]

            batch_size = 32
            embeddings_list = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.model.encode(batch, show_progress_bar=False)
                embeddings_list.append(batch_embeddings)

            self.embeddings = np.vstack(embeddings_list)

            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)

            print(f"Generated and saved embeddings for {len(texts)} clean chunks with shape: {self.embeddings.shape}")

        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            print(traceback.format_exc())
            self.embeddings = None

    def retrieve(self, query: str, top_k: int = 3, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant clean chunks for a query (top_k defaults to 3)

        Args:
            query: Search query
            top_k: Number of top results to return (default: 3)
            category_filter: Optional category to pre-filter by

        Returns:
            List of top clean chunks with similarity scores
        """
        try:
            if not self.chunks:
                print("No clean chunks available.")
                return []

            similarities = None
            try:
                if self.embeddings is not None:
                    self.load_model()
                    if self.model is not None:
                        query_embedding = self.model.encode([query])
                        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            except Exception as load_err:
                print(f"Notice: Vector model encoding unavailable ({load_err}). Using keyword similarity fallback.")
                similarities = None

            if similarities is None:
                query_words = set(query.lower().split())
                similarities = []
                for chunk in self.chunks:
                    text_words = set(chunk.get('text', '').lower().split())
                    if not query_words or not text_words:
                        sim = 0.0
                    else:
                        overlap = query_words.intersection(text_words)
                        sim = len(overlap) / len(query_words)
                    similarities.append(sim)

            candidate_results = []
            for i, similarity in enumerate(similarities):
                chunk = self.chunks[i].copy()

                if category_filter:
                    chunk_category = chunk.get('category', '').lower()
                    if chunk_category != category_filter.lower():
                        continue

                chunk['similarity_score'] = float(similarity)
                candidate_results.append((i, similarity, chunk))

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

            # Debug log top chunk for verification
            if top_results:
                top_c = top_results[0]
                print("TOP CHUNK:", top_c['text'][:200])

            return top_results

        except Exception as e:
            print(f"Error in retrieve method: {str(e)}")
            return []

    def hybrid_retrieve(self, query: str, top_k: int = 3,
                        category_filter: Optional[str] = None,
                        boost_recent: bool = False) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval returning top 3 clean chunks with metadata boosting.
        """
        try:
            results = self.retrieve(query, top_k * 5, category_filter)

            if not results:
                print("No candidates found for hybrid retrieval.")
                return []

            for result in results:
                hybrid_score = result['similarity_score']

                if result['word_count'] > 300:
                    hybrid_score *= 1.05

                query_lower = query.lower()
                title_lower = result.get('title', '').lower()
                section_lower = result.get('section', '').lower()

                if query_lower in title_lower:
                    hybrid_score *= 1.2
                elif any(word in title_lower for word in query_lower.split()):
                    hybrid_score *= 1.1

                if query_lower in section_lower:
                    hybrid_score *= 1.15

                result['hybrid_score'] = float(hybrid_score)

            results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            top_clean = results[:top_k]

            if top_clean:
                top_c = top_clean[0]
                print("TOP CHUNK:", top_c['text'][:200])

            return top_clean

        except Exception as e:
            print(f"Error in hybrid_retrieve method: {str(e)}")
            print(traceback.format_exc())
            return []

    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the clean embedding space
        """
        if self.embeddings is None:
            return {"error": "No embeddings available"}

        return {
            "total_chunks": len(self.chunks),
            "embedding_dimension": self.embeddings.shape[1],
            "model_name": self.model_name,
            "embedding_file_exists": os.path.exists(self.embeddings_file)
        }