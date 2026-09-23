import re
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import pypdf

def clean_text(text: str) -> str:
    """Clean text by removing extra spaces and converting to lowercase."""
    if not text:
        return ""
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip().lower()

def split_into_chunks(text: str, chunk_size: int = 125, overlap: int = 20) -> list[str]:
    """Split text into chunks of 100-150 words (default ~125 words per chunk)."""
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text:
            chunks.append(chunk_text)
        i += step
    return chunks

def extract_text_from_file(uploaded_file) -> str:
    """Extract raw text from uploaded PDF or TXT file."""
    if uploaded_file is None:
        return ""
    try:
        filename = uploaded_file.name.lower()
        if filename.endswith(".pdf"):
            reader = pypdf.PdfReader(uploaded_file)
            extracted_pages = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    extracted_pages.append(t)
            return "\n".join(extracted_pages)
        elif filename.endswith(".txt"):
            content = uploaded_file.getvalue()
            return content.decode("utf-8", errors="ignore")
        else:
            return ""
    except Exception:
        return ""

@st.cache_resource
def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """Cache the sentence transformer model resource."""
    return SentenceTransformer(model_name)

class VectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.chunks: list[str] = []
        self.embeddings: np.ndarray | None = None

    @property
    def model(self):
        return load_embedding_model(self.model_name)

    def add_document(self, raw_text: str) -> int:
        """Process raw text, clean, chunk (100-150 words), generate embeddings and store them."""
        cleaned = clean_text(raw_text)
        if not cleaned:
            self.chunks = []
            self.embeddings = None
            return 0
        
        new_chunks = split_into_chunks(cleaned, chunk_size=125, overlap=20)
        if not new_chunks:
            self.chunks = []
            self.embeddings = None
            return 0
        
        self.chunks = new_chunks
        self.embeddings = self.model.encode(self.chunks, convert_to_numpy=True)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 3) -> tuple[list[dict], float]:
        """Perform cosine similarity search and return top k chunks with similarity scores."""
        if not self.chunks or self.embeddings is None or len(self.embeddings) == 0:
            return [], 0.0

        cleaned_query = clean_text(query)
        if not cleaned_query:
            return [], 0.0

        query_embedding = self.model.encode([cleaned_query], convert_to_numpy=True)[0]

        norm_query = np.linalg.norm(query_embedding)
        norm_embeds = np.linalg.norm(self.embeddings, axis=1)

        denom = norm_query * norm_embeds
        denom = np.where(denom == 0, 1e-10, denom)

        cosine_sims = np.dot(self.embeddings, query_embedding) / denom

        top_indices = np.argsort(cosine_sims)[::-1][:top_k]

        max_score = float(cosine_sims[top_indices[0]]) if len(top_indices) > 0 else 0.0
        max_score = float(np.clip(max_score, 0.0, 1.0))

        retrieved = []
        for idx in top_indices:
            score = float(cosine_sims[idx])
            score = float(np.clip(score, 0.0, 1.0))
            retrieved.append({
                "text": self.chunks[idx],
                "score": score
            })

        return retrieved, max_score
