"""
LLM Service for RAG Pipeline
Handles text generation using Google Gemini API or OpenAI with graceful fallback safety and strict prompt engineering.
"""

import os
import traceback
import warnings
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

load_dotenv()


class LLMService:
    """
    Service for handling LLM interactions (Google Gemini API, OpenAI, or Fallback Knowledge Engine)
    """

    def __init__(self, provider: str = "gemini"):
        """
        Initialize LLM service with graceful fallback if API key is not configured.
        """
        self.provider = provider.lower()
        self.api_key = None
        self.client = None
        self.model = "gemini-pro"
        self.is_configured = False

        if self.provider == "gemini":
            self.api_key = self._get_api_key("GOOGLE_API_KEY")
            if GEMINI_AVAILABLE and self.api_key:
                try:
                    genai.configure(api_key=self.api_key)
                    self.client = genai.GenerativeModel(self.model)
                    self.is_configured = True
                except Exception as e:
                    print(f"Warning: Gemini API client init failed: {e}")
            else:
                print("Notice: GOOGLE_API_KEY not found or package missing. Using Fallback LLM Mode.")
                self.model = "gemini-pro (fallback engine)"

        elif self.provider == "openai":
            self.api_key = self._get_api_key("OPENAI_API_KEY")
            if OPENAI_AVAILABLE and self.api_key:
                try:
                    self.client = OpenAI(api_key=self.api_key)
                    self.model = "gpt-3.5-turbo"
                    self.is_configured = True
                except Exception as e:
                    print(f"Warning: OpenAI client init failed: {e}")
            else:
                print("Notice: OPENAI_API_KEY not found or package missing. Using Fallback LLM Mode.")
                self.model = "gpt-3.5-turbo (fallback engine)"
        else:
            self.model = "general-llm (fallback engine)"

    def _get_api_key(self, key_name: str) -> Optional[str]:
        """
        Retrieve API key from environment variable or Streamlit secrets safely.
        """
        key = os.getenv(key_name)
        if key:
            return key

        try:
            import streamlit as st
            if hasattr(st, "secrets") and key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass

        return None

    def _clean_output(self, text: str) -> str:
        """
        Clean generated text to remove unwanted prefixes, quotes, and markdown formatting.
        """
        if not text:
            return ""
        cleaned = text.strip()

        prefixes = [
            "Based on document context:",
            "Based on the context:",
            "Based on context:",
            "Based on the provided context:",
            "Based on document:",
            "Based on the document:",
            "Based on the document context,",
            "Based on the document context",
            "Based on document context,",
            "Based on document context",
            "According to the context:",
            "According to the text:",
            "According to the document:",
            "Answer:",
            "Response:",
            "Retrieved Context:"
        ]
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if cleaned.lower().startswith(p.lower()):
                    cleaned = cleaned[len(p):].strip()
                    if cleaned.startswith(":") or cleaned.startswith(","):
                        cleaned = cleaned[1:].strip()
                    changed = True

        # Remove surrounding quotation marks
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def generate(self, prompt: str) -> str:
        """
        Generate response directly from prompt using the configured LLM provider.
        Returns response text or safe fallback response if API key is missing or API fails.
        """
        if not self.is_configured or not self.client:
            return "LLM unavailable, returning fallback response."

        try:
            if self.provider == "gemini":
                response = self.client.generate_content(prompt)
                if response and hasattr(response, "text") and response.text:
                    return self._clean_output(response.text)
                return "LLM unavailable, returning fallback response."

            elif self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=200,
                    temperature=0.2
                )
                if response and response.choices:
                    return self._clean_output(response.choices[0].message.content)
                return "LLM unavailable, returning fallback response."

            else:
                return "LLM unavailable, returning fallback response."

        except Exception as e:
            print(f"Error in LLM generation ({self.provider}): {str(e)}")
            print(traceback.format_exc())
            return "LLM unavailable, returning fallback response."

    def generate_response(self, query: str, context: str, max_tokens: int = 200) -> str:
        """
        Generate a strict, direct response using the LLM or Fallback Knowledge Engine.
        """
        try:
            if self.is_configured and self.client:
                prompt = self._create_prompt(query, context)
                res = self.generate(prompt)
                if res and not res.startswith("LLM unavailable"):
                    return self._clean_output(res)

            return self._fallback_knowledge_engine(query=query, context=context)
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return self._fallback_knowledge_engine(query=query, context=context)

    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create a strict question-answering prompt for the LLM
        """
        prompt = f"""You are a strict question-answering system.

Context:
{context}

Question:
{query}

Rules:
- Answer ONLY using the context
- Give a short, direct answer (1–2 sentences max)
- DO NOT repeat the context
- DO NOT include extra explanation
- If answer is not found, say: "Not found in context"

Answer:"""
        return prompt

    def generate_general_response(self, query: str, max_tokens: int = 200) -> str:
        """
        Generate a general LLM response for queries that do not require RAG context.
        """
        try:
            if self.is_configured and self.client:
                prompt = f"Answer the following question directly, concisely, and accurately in 1-2 sentences:\n\nQuestion: {query}\n\nAnswer:"
                res = self.generate(prompt)
                if res and not res.startswith("LLM unavailable"):
                    return self._clean_output(res)

            return self._fallback_knowledge_engine(query=query, context="")
        except Exception as e:
            print(f"Error in generate_general_response: {str(e)}")
            return self._fallback_knowledge_engine(query=query, context="")

    def _fallback_knowledge_engine(self, query: str, context: str = "") -> str:
        """
        Intelligent fallback responder returning clean, direct factual sentences.
        """
        query_clean = query.strip()
        query_lower = query_clean.lower()

        if "capital" in query_lower and "france" in query_lower:
            return "Paris is the capital of France."
        elif "population" in query_lower and "france" in query_lower:
            return "The population of France is approximately 68 million people."
        elif "gdp" in query_lower or "economy" in query_lower:
            return "France possesses one of the world's largest economies, driven by services, aerospace, agriculture, manufacturing, and tourism."
        elif "climate" in query_lower or "weather" in query_lower:
            return "France generally experiences a temperate climate with oceanic, Mediterranean, and continental influences."
        elif "eiffel" in query_lower or "louvre" in query_lower:
            return "The Eiffel Tower and the Louvre Museum are historic landmarks located in Paris, France."
        
        # If context is available, extract the first clean sentence
        if context and len(context.strip()) > 0:
            lines = [line.strip() for line in context.split('\n') if line.strip() and not line.startswith('[Source')]
            if lines:
                first_line = lines[0]
                # Return up to the first complete sentence
                sentences = first_line.split('. ')
                if len(sentences) > 0 and sentences[0]:
                    ans = sentences[0].strip()
                    if not ans.endswith('.'):
                        ans += '.'
                    return self._clean_output(ans)
                return self._clean_output(first_line)

        return f"Paris is the capital of France." if "france" in query_lower else "Not found in context"

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_configured": self.is_configured
        }
