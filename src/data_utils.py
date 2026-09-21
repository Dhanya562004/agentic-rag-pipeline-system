"""
Data Processing Module for RAG Pipeline
Handles web scraping, text cleaning, chunking, and metadata extraction
"""

import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import json
from typing import List, Dict, Any
from urllib.parse import urljoin, urlparse
import time
import os
import unicodedata

class DataProcessor:
    """Handles data collection and processing for the RAG pipeline"""
    
    def __init__(self):
        self.base_url = "https://www.britannica.com"
        self.sections = [
            "https://www.britannica.com/place/France/Land",
            "https://www.britannica.com/place/France/The-Hercynian-massifs",
            "https://www.britannica.com/place/France/The-great-lowlands",
            "https://www.britannica.com/place/France/The-younger-mountains-and-adjacent-plains",
            "https://www.britannica.com/place/France/Drainage",
            "https://www.britannica.com/place/France/Soils",
            "https://www.britannica.com/place/France/Climate",
            "https://www.britannica.com/place/France/Plant-and-animal-life"
        ]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape content from a single URL
        
        Args:
            url: The URL to scrape
            
        Returns:
            Dictionary containing scraped data with metadata
        """
        try:
            print(f"Scraping: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_element = soup.find('h1') or soup.find('title')
            title = title_element.get_text().strip() if title_element else "Unknown Title"
            
            # Extract content with section information
            content_data = self._extract_content_with_sections(soup)
            
            # Extract section/category from URL
            url_parts = urlparse(url).path.split('/')
            category = url_parts[-1] if len(url_parts) > 1 else "general"
            
            return {
                'url': url,
                'title': title,
                'content': content_data['text'],
                'sections': content_data['sections'],  # NEW: Section mappings
                'category': category,
                'word_count': len(content_data['text'].split()),
                'scraped_at': datetime.now().strftime('%Y-%m-%d')
            }
            
        except Exception as e:
            print(f"Error scraping {url}: {str(e)}")
            return {
                'url': url,
                'title': "Error",
                'content': "",
                'sections': [],
                'category': "error",
                'word_count': 0,
                'scraped_at': datetime.now().strftime('%Y-%m-%d'),
                'error': str(e)
            }

    def _extract_content_with_sections(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """
        Extract content while preserving section structure
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Dictionary with text and section mappings
        """
        content_selectors = [
            'div.content',
            'div.article-content', 
            'div.entry-content',
            'main',
            'article',
            '.md-def-body',
            '.topic-content'
        ]
        
        content_container = None
        for selector in content_selectors:
            container = soup.select_one(selector)
            if container:
                content_container = container
                break
        
        if not content_container:
            # Fallback to all paragraphs
            paragraphs = soup.find_all('p')
            full_text = "\n".join([p.get_text() for p in paragraphs])
            return {
                'text': full_text,
                'sections': [{'start': 0, 'end': len(full_text), 'heading': 'Main Content'}]
            }
        
        # Extract content with section boundaries
        sections = []
        full_text = ""
        current_section = "Introduction"
        section_start = 0
        
        for element in content_container.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div']):
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Save previous section
                if full_text.strip():
                    sections.append({
                        'start': section_start,
                        'end': len(full_text),
                        'heading': current_section
                    })
                
                # Start new section
                current_section = element.get_text().strip()
                section_start = len(full_text)
                
            elif element.name in ['p', 'div']:
                # Add paragraph content
                text = element.get_text()
                if text.strip():
                    full_text += text + "\n"
        
        # Add final section
        if full_text.strip():
            sections.append({
                'start': section_start,
                'end': len(full_text),
                'heading': current_section
            })
        
        return {
            'text': full_text.strip(),
            'sections': sections
        }
    
    def scrape_all_sections(self) -> List[Dict[str, Any]]:
        """
        Scrape all defined sections
        
        Returns:
            List of dictionaries containing scraped data
        """
        all_data = []
        
        for url in self.sections:
            data = self.scrape_url(url)
            if data['content']:
                all_data.append(data)
            
            time.sleep(1)
        
        return all_data
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Fix encoding issues - normalize unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Remove or replace common HTML entities that might have been missed
        html_entities = {
            '&nbsp;': ' ',
            '&amp;': '&',
            '&lt;': '<',
            '&gt;': '>',
            '&quot;': '"',
            '&#39;': "'",
            '&apos;': "'",
            '&rsquo;': "'",
            '&lsquo;': "'",
            '&rdquo;': '"',
            '&ldquo;': '"',
            '&mdash;': '—',
            '&ndash;': '–'
        }
        
        for entity, replacement in html_entities.items():
            text = text.replace(entity, replacement)
        
        # Remove extra whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\r+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        # Remove or replace problematic characters - fix quote normalization
        # Handle different types of quotes individually to avoid regex parsing issues
        text = text.replace('"', '"')  # Left double quotation mark
        text = text.replace('"', '"')  # Right double quotation mark
        text = text.replace(''', "'")  # Left single quotation mark
        text = text.replace(''', "'")  # Right single quotation mark
        text = text.replace('„', '"')  # Double low-9 quotation mark
        text = text.replace('‚', "'")  # Single low-9 quotation mark
        
        text = re.sub(r'[–—]', '-', text)  # Normalize dashes
        text = re.sub(r'[…]', '...', text)  # Replace ellipsis
        
        # Remove special characters but keep essential punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\%\$\&\@\#]', '', text)
        
        # Fix multiple punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\,{2,}', ',', text)
        text = re.sub(r'\!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Remove URLs and email addresses (noise)
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove standalone numbers that might be page numbers or references
        text = re.sub(r'\b\d{4,}\b', '', text)  # Remove 4+ digit numbers (likely page refs)
        
        # Clean up citation patterns like [1], (2), etc.
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\(\d+\)', '', text)
        
        # Remove excessive spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Strip leading/trailing whitespace and ensure single spaces
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 2, method: str = "semantic") -> List[str]:
        """
        Split text into overlapping chunks using different strategies
        
        Args:
            text: Text to chunk
            chunk_size: Maximum tokens per chunk (for semantic/sentence methods)
            overlap: Number of sentences to overlap (for semantic/sentence methods) 
            method: 'fixed' (old way), 'sentences', or 'semantic' (recommended)
            
        Returns:
            List of text chunks
        """
        if not text:
            return []
        
        if method == "fixed":
            # Original fixed-length word chunking (kept for comparison)
            return self._chunk_text_fixed_length(text, chunk_size, overlap)
        elif method == "sentences":
            # Sentence-aware chunking
            return self.chunk_text_by_sentences(text, chunk_size, overlap)
        elif method == "semantic":
            # Paragraph + sentence aware chunking (recommended)
            return self.chunk_text_semantic_aware(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking method: {method}")

    def _chunk_text_fixed_length(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Original fixed-length chunking method"""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunks.append(' '.join(chunk_words))
            
            if end >= len(words):
                break
                
            start = end - overlap
        
        return chunks

    def chunk_text_by_sentences(self, text: str, max_tokens: int = 400, overlap_sentences: int = 2) -> List[str]:
        """
        Split text into chunks by sentences with overlap
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk (roughly 300-500 words)
            overlap_sentences: Number of sentences to overlap between chunks
            
        Returns:
            List of coherent text chunks
        """
        if not text:
            return []
        
        # Split on sentence boundaries while preserving the delimiter
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        if len(sentences) <= 3:  # Very short text
            return [text]
        
        chunks = []
        current_chunk_sentences = []
        current_token_count = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = len(sentence.split())
            
            # If adding this sentence would exceed max_tokens, finalize current chunk
            if current_token_count + sentence_tokens > max_tokens and current_chunk_sentences:
                # Create chunk from current sentences
                chunk_text = ' '.join(current_chunk_sentences)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                if len(current_chunk_sentences) >= overlap_sentences:
                    # Keep last few sentences for overlap
                    overlap_start = len(current_chunk_sentences) - overlap_sentences
                    current_chunk_sentences = current_chunk_sentences[overlap_start:]
                    current_token_count = sum(len(s.split()) for s in current_chunk_sentences)
                else:
                    current_chunk_sentences = []
                    current_token_count = 0
            
            # Add current sentence
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            i += 1
        
        # Add final chunk
        if current_chunk_sentences:
            chunk_text = ' '.join(current_chunk_sentences)
            chunks.append(chunk_text)
        
        return chunks

    def chunk_text_semantic_aware(self, text: str, max_tokens: int = 400, overlap_sentences: int = 2) -> List[str]:
        """
        Enhanced chunking that respects paragraph boundaries and semantic structure
        
        Args:
            text: Text to chunk
            max_tokens: Maximum tokens per chunk
            overlap_sentences: Number of sentences to overlap
            
        Returns:
            List of semantically coherent chunks
        """
        if not text:
            return []
        
        # First, split by paragraphs (double newlines or clear breaks)
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        if not paragraphs:
            return self.chunk_text_by_sentences(text, max_tokens, overlap_sentences)
        
        chunks = []
        current_chunk_text = ""
        current_token_count = 0
        
        for paragraph in paragraphs:
            paragraph_tokens = len(paragraph.split())
            
            # If this paragraph alone exceeds max_tokens, chunk it by sentences
            if paragraph_tokens > max_tokens:
                # Finalize current chunk if it exists
                if current_chunk_text:
                    chunks.append(current_chunk_text.strip())
                    current_chunk_text = ""
                    current_token_count = 0
                
                # Chunk the large paragraph by sentences
                para_chunks = self.chunk_text_by_sentences(paragraph, max_tokens, overlap_sentences)
                chunks.extend(para_chunks)
                continue
            
            # If adding this paragraph would exceed max_tokens, finalize current chunk
            if current_token_count + paragraph_tokens > max_tokens and current_chunk_text:
                chunks.append(current_chunk_text.strip())
                current_chunk_text = paragraph
                current_token_count = paragraph_tokens
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n" + paragraph
                else:
                    current_chunk_text = paragraph
                current_token_count += paragraph_tokens
        
        # Add final chunk
        if current_chunk_text:
            chunks.append(current_chunk_text.strip())
        
        return chunks
    def process_scraped_data(self, raw_data: List[Dict[str, Any]], 
                       chunk_size: int = 400, overlap: int = 2, method: str = "semantic") -> List[Dict[str, Any]]:
        """
        Process scraped data: clean text, create chunks, add metadata
        
        Args:
            raw_data: List of scraped data dictionaries
            chunk_size: Maximum tokens per chunk (for semantic/sentence methods)
            overlap: Number of sentences to overlap (for semantic/sentence methods)
            method: Chunking method - 'fixed', 'sentences', or 'semantic' (recommended)
            
        Returns:
            List of processed chunks with metadata
        """
        processed_chunks = []
        
        for doc_idx, doc in enumerate(raw_data):
            if not doc.get('content'):
                continue
                
            # Clean the content
            cleaned_content = self.clean_text(doc['content'])
            
            if not cleaned_content:
                continue
            # Create chunks using enhanced semantic chunking
            chunks = self.chunk_text(cleaned_content, chunk_size, overlap, method=method)

            # Add metadata to each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Find which section this chunk belongs to
                chunk_position = self._find_chunk_position_in_original(
                    doc['content'], chunk, doc.get('sections', [])
                )
                
                processed_chunk = {
                    'chunk_id': f"doc_{doc_idx}_chunk_{chunk_idx}",
                    'text': chunk,
                    'source_url': doc['url'],
                    'title': doc['title'],
                    'section': chunk_position['section'],  # NEW: Granular section
                    'category': doc['category'],
                    'doc_index': doc_idx,
                    'chunk_index': chunk_idx,
                    'total_chunks': len(chunks),
                    'word_count': len(chunk.split()),
                    'date_retrieved': doc['scraped_at']  # Renamed for clarity
                }
                processed_chunks.append(processed_chunk)
        
        return processed_chunks

    def _find_chunk_position_in_original(self, original_text: str, chunk: str, 
                                   sections: List[Dict]) -> Dict[str, str]:
        """
        Find which section a chunk belongs to
        
        Args:
            original_text: Original document text
            chunk: Text chunk to locate
            sections: List of section boundaries
            
        Returns:
            Dictionary with section information
        """
        if not sections:
            return {'section': 'Main Content'}
        
        # Find approximate position of chunk in original text
        chunk_start = original_text.find(chunk[:50])  # Use first 50 chars to locate
        
        if chunk_start == -1:
            return {'section': 'Unknown Section'}
        
        # Find which section contains this position
        for section in sections:
            if section['start'] <= chunk_start <= section['end']:
                return {'section': section['heading']}
        
        return {'section': 'Main Content'}
    
    def save_data(self, data: List[Dict[str, Any]], filename: str = "processed_data.json", remove_duplicates: bool = True):
        """
        Save processed data to JSON file
        
        Args:
            data: Processed data to save
            filename: Output filename
            remove_duplicates: Whether to remove duplicate chunks before saving
        """
        # Ensure the data directory exists
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_dir = os.path.abspath(data_dir)
        os.makedirs(data_dir, exist_ok=True)

        # Always save/load from the data directory
        file_path = os.path.join(data_dir, filename)

        if remove_duplicates:
            # Create a dictionary to track seen texts
            seen_texts = {}
            unique_chunks = []
            duplicates_removed = 0
            
            for chunk in data:
                # Create a content hash from the first 100 characters (enough to identify duplicates)
                content_hash = hash(chunk['text'][:100])
                
                if content_hash not in seen_texts:
                    seen_texts[content_hash] = True
                    unique_chunks.append(chunk)
                else:
                    duplicates_removed += 1
            
            if duplicates_removed > 0:
                print(f"Removed {duplicates_removed} duplicate chunks")
            
            # Update chunk IDs to be sequential
            for i, chunk in enumerate(unique_chunks):
                doc_index = chunk['doc_index']
                chunk['chunk_id'] = f"doc_{doc_index}_chunk_{i}"
                chunk['chunk_index'] = i
            
            data = unique_chunks
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(data)} chunks to {file_path}")
    
    def load_data(self, filename: str = "processed_data.json") -> List[Dict[str, Any]]:
        """
        Load processed data from JSON file
        
        Args:
            filename: Input filename
            
        Returns:
            List of processed chunks
        """
        # Always load from the data directory
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        data_dir = os.path.abspath(data_dir)
        file_path = os.path.join(data_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} chunks from {file_path}")
            return data
        except FileNotFoundError:
            print(f"File {file_path} not found")
            return []
    
    def get_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about the processed data
        
        Args:
            data: Processed data
            
        Returns:
            Dictionary with statistics
        """
        if not data:
            return {}
        
        total_chunks = len(data)
        total_words = sum(chunk['word_count'] for chunk in data)
        categories = list(set(chunk['category'] for chunk in data))
        avg_words_per_chunk = total_words / total_chunks if total_chunks > 0 else 0
        category_counts = {}
        section_counts = {}  # NEW: Track section distribution
        for chunk in data:
            cat = chunk['category']
            category_counts[cat] = category_counts.get(cat, 0) + 1
            
            # Track sections if available
            section = chunk.get('section', 'Unknown')
            section_counts[section] = section_counts.get(section, 0) + 1
        
        return {
            'total_chunks': total_chunks,
            'total_words': total_words,
            'average_words_per_chunk': round(avg_words_per_chunk, 2),
            'categories': categories,
            'category_counts': category_counts,
            'section_counts': section_counts,  # NEW: Section distribution
            'unique_sources': len(set(chunk['source_url'] for chunk in data))
        }


def main():
    """Main function to run data collection and processing"""
    processor = DataProcessor()
    
    print("Starting data collection...")
    

    # Scrape all sections
    raw_data = processor.scrape_all_sections()
    print(f"Scraped {len(raw_data)} documents")

    # Process the data using enhanced semantic chunking
    print("Processing and chunking data with semantic-aware method...")
    processed_data = processor.process_scraped_data(
        raw_data, 
        chunk_size=400,         # 400 tokens per chunk
        overlap=2,              # 2 sentences overlap
        method="semantic"       # Use semantic-aware chunking
    )
    
    processor.save_data(processed_data)
    data = processed_data
    
    # Print statistics
    stats = processor.get_data_statistics(data)
    print("\nData Statistics:")
    print(f"Total chunks: {stats['total_chunks']}")
    print(f"Total words: {stats['total_words']}")
    print(f"Average words per chunk: {stats['average_words_per_chunk']}")
    print(f"Categories: {stats['categories']}")
    print(f"Category distribution: {stats['category_counts']}")
    print(f"Unique sources: {stats['unique_sources']}")
    
    return data


if __name__ == "__main__":
    main()
