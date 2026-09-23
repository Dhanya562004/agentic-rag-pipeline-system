"""
🔥 FINAL LLM SERVICE (STABLE + DEPLOYABLE)

- Uses google-generativeai (correct package)
- Works with Streamlit secrets
- Handles errors properly
- Supports RAG + fallback
"""

import streamlit as st
import google.generativeai as genai


class LLMService:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.model = None

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-pro")
                print("✅ Gemini initialized successfully")
            except Exception as e:
                print("❌ Gemini init error:", str(e))
        else:
            print("⚠️ API key not found")

    # -------------------------
    # GET API KEY (STREAMLIT SECRETS)
    # -------------------------
    def _get_api_key(self):
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except Exception:
            return None

    # -------------------------
    # MAIN FUNCTION (USED BY PIPELINE)
    # -------------------------
    def generate_response(self, prompt: str, context: str = None) -> str:

        if not self.api_key:
            return "API key not configured"

        if not self.model:
            return "Model not initialized"

        try:
            # -------------------------
            # RAG MODE
            # -------------------------
            if context and context.strip():
                final_prompt = f"""
You are a helpful AI assistant.

Use ONLY the given context to answer.

Context:
{context}

Question:
{prompt}

Rules:
- Keep answer short (2-3 lines)
- If answer not in context, say "Not found in document"

Answer:
"""
            else:
                # -------------------------
                # GENERAL MODE
                # -------------------------
                final_prompt = f"""
You are a helpful AI assistant.

Answer clearly and simply.

Question:
{prompt}

Answer:
"""

            response = self.model.generate_content(final_prompt)

            # Safe extraction
            if hasattr(response, "text") and response.text:
                return response.text.strip()

            return "⚠️ Empty response from AI"

        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"

    # -------------------------
    # OPTIONAL INFO
    # -------------------------
    def get_model_info(self):
        return {
            "api_key_loaded": bool(self.api_key),
            "model_loaded": self.model is not None
        }
