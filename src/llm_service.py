import streamlit as st
import google.generativeai as genai


class LLMService:
    def __init__(self, model_name="gemini-1.5-flash"):
        self.model_name = model_name
        self.model = None

        api_key = self._get_api_key()

        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print("✅ Gemini initialized successfully")
            except Exception as e:
                print("❌ Gemini init error:", str(e))
        else:
            print("❌ API key not found")

    # -----------------------
    # GET API KEY
    # -----------------------
    def _get_api_key(self):
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except:
            return None

    # -----------------------
    # MAIN GENERATE FUNCTION
    # -----------------------
    def generate_response(self, prompt="", context=None):
        if not self.model:
            return "❌ API key not configured"

        try:
            if context:
                full_prompt = f"""
You are a helpful AI assistant.

Use ONLY the context below to answer.

Context:
{context}

Question:
{prompt}

Answer:
"""
            else:
                full_prompt = prompt

            response = self.model.generate_content(full_prompt)

            if hasattr(response, "text") and response.text:
                return response.text.strip()

            return "⚠️ Empty response from AI"

        except Exception as e:
            return f"❌ Gemini Error: {str(e)}"
