import streamlit as st
import google.generativeai as genai

class LLMService:
    def __init__(self, model_name="gemini-pro"):
        self.model_name = model_name
        self.model = None

        api_key = self._get_api_key()

        if api_key:
            try:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print("✅ Gemini initialized")
            except Exception as e:
                print("❌ Gemini init error:", str(e))
        else:
            print("❌ API key not found")

    # -------------------------
    # GET API KEY (STREAMLIT CLOUD)
    # -------------------------
    def _get_api_key(self):
        try:
            return st.secrets["GOOGLE_API_KEY"]
        except:
            return None

    # -------------------------
    # MAIN GENERATE FUNCTION
    # -------------------------
    def generate_response(self, prompt="", context=None):
        if not self.model:
            return "API key not configured"

        try:
            if context:
                full_prompt = f"""
Use the following context to answer:

{context}

Question: {prompt}
Answer:
"""
            else:
                full_prompt = prompt

            response = self.model.generate_content(full_prompt)

            if hasattr(response, "text") and response.text:
                return response.text.strip()

            return "No response generated"

        except Exception as e:
            print("❌ Gemini error:", str(e))
            return f"Error: {str(e)}"
