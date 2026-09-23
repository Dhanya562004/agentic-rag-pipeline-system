import os
import streamlit as st
import google.generativeai as genai


class LLMService:
    def __init__(self):
        self.model = None
        api_key = self._get_api_key()

        print("🔑 API KEY FOUND:", bool(api_key))

        if api_key:
            try:
                genai.configure(api_key=api_key)

                # ✅ ONLY WORKING MODEL FOR THIS SDK
                self.model = genai.GenerativeModel("gemini-pro")

                print("✅ Gemini initialized successfully")

            except Exception as e:
                print("❌ Gemini init error:", str(e))
        else:
            print("❌ No API key found")

    # -------------------------
    def _get_api_key(self):
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                return st.secrets["GOOGLE_API_KEY"]
        except:
            pass

        return os.getenv("GOOGLE_API_KEY")

    # -------------------------
    def generate_response(self, prompt, context=None):

        if not self.model:
            return "❌ API key not configured"

        try:
            if context:
                full_prompt = f"""
Answer ONLY using the context.

Context:
{context}

Question:
{prompt}
"""
            else:
                full_prompt = prompt

            response = self.model.generate_content(full_prompt)

            if response and hasattr(response, "text"):
                return response.text

            return "⚠️ Empty response"

        except Exception as e:
            print("❌ Gemini error:", str(e))
            return f"❌ Gemini Error: {str(e)}"
