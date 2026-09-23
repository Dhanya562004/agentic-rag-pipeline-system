import streamlit as st

class LLMService:
    def __init__(self):
        self.api_key = self._get_api_key()

        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.available = True
                print("✅ Gemini initialized")
            except Exception as e:
                print("❌ Gemini init error:", e)
                self.available = False
        else:
            print("❌ API KEY NOT FOUND")
            self.available = False

    def _get_api_key(self):
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except:
            return None

    def generate_response(self, prompt: str, context: str = None):
        if not self.available:
            return "API key not configured or model failed."

        try:
            if context:
                final_prompt = f"""
Answer using ONLY the context.

Context:
{context}

Question:
{prompt}

Answer:
"""
            else:
                final_prompt = prompt

            response = self.model.generate_content(final_prompt)

            if hasattr(response, "text") and response.text:
                return response.text.strip()

            return "⚠️ Empty response"

        except Exception as e:
            print("❌ Gemini Error:", e)
            return f"❌ Gemini Error: {str(e)}"
