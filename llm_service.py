import streamlit as st

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

        if context and context.strip():
            full_prompt = (
                f"You are a helpful AI assistant. Use the following context to answer the user's question accurately.\n\n"
                f"--- CONTEXT ---\n{context}\n---------------\n\n"
                f"Question: {prompt}\n\nAnswer:"
            )
        else:
            full_prompt = prompt

        # 1. Primary: Try official google-genai SDK
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            models_to_try = [self.model_name, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
            
            for m in models_to_try:
                try:
                    response = client.models.generate_content(
                        model=m,
                        contents=full_prompt,
                    )
                    if response and hasattr(response, "text") and response.text:
                        return response.text.strip()
                except Exception as inner_e:
                    err_text = str(inner_e).lower()
                    if "404" in err_text or "not found" in err_text:
                        continue
                    if "api_key" in err_text or "unauthenticated" in err_text or "blocked" in err_text:
                        return "API key not configured or blocked"
                    continue
        except Exception as e1:
            err_text = str(e1).lower()
            if "api_key" in err_text or "unauthenticated" in err_text or "blocked" in err_text:
                return "API key not configured or blocked"

        # 2. Backup: Try legacy google-generativeai SDK
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            legacy_models = [self.model_name, "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
            
            for m in legacy_models:
                try:
                    model = legacy_genai.GenerativeModel(m)
                    res = model.generate_content(full_prompt)
                    if res and hasattr(res, "text") and res.text:
                        return res.text.strip()
                except Exception:
                    continue
        except Exception:
            pass

        return "AI service temporarily unavailable"

    def generate_general_response(self, prompt: str) -> str:
        return self.generate_response(prompt=prompt, context=None)
