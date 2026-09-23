import streamlit as st
from google import genai

class LLMService:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    def _get_api_key(self) -> str | None:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                key = st.secrets["GOOGLE_API_KEY"]
                if key and str(key).strip():
                    return str(key).strip()
        except Exception:
            pass
        return None

    def generate_response(self, prompt: str = "", context: str | None = None, **kwargs) -> str:
        if not prompt and "query" in kwargs:
            prompt = kwargs["query"]
        api_key = self._get_api_key()
        if not api_key:
            return "API key not configured"

        try:
            client = genai.Client(api_key=api_key)
            if context and context.strip():
                full_prompt = (
                    f"You are a helpful AI assistant. Use the following context to answer the user's question accurately.\n\n"
                    f"--- CONTEXT ---\n{context}\n---------------\n\n"
                    f"Question: {prompt}\n\nAnswer:"
                )
            else:
                full_prompt = prompt

            response = client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
            )
            if response and hasattr(response, "text") and response.text:
                return response.text.strip()
            else:
                return "AI service temporarily unavailable"
        except Exception:
            return "AI service temporarily unavailable"
