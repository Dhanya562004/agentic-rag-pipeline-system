import streamlit as st

class LLMService:
    def __init__(self, model_name: str = "gemini-1.5-flash"):
        self.model_name = model_name

    def _get_api_key(self) -> str | None:
        try:
            if "GOOGLE_API_KEY" in st.secrets:
                raw_key = str(st.secrets["GOOGLE_API_KEY"])
                key = raw_key.strip().strip('"').strip("'")
                if key:
                    return key
        except Exception:
            pass
        return None

    def _extract_text(self, response) -> str | None:
        if not response:
            return None
        try:
            if hasattr(response, "text") and response.text:
                return response.text.strip()
        except Exception:
            pass
        try:
            if hasattr(response, "candidates") and response.candidates:
                for cand in response.candidates:
                    if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                        parts = [p.text for p in cand.content.parts if hasattr(p, "text") and p.text]
                        if parts:
                            return "".join(parts).strip()
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

        errors = []

        # 1. Primary: Try modern google-genai SDK
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
                    text = self._extract_text(response)
                    if text:
                        return text
                except Exception as inner_e:
                    err_msg = str(inner_e)
                    errors.append(f"genai({m}): {err_msg}")
                    err_text = err_msg.lower()
                    if "404" in err_text or "not found" in err_text:
                        continue
                    break
        except Exception as e1:
            errors.append(f"genai_init: {str(e1)}")

        # 2. Backup: Try legacy google-generativeai SDK
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            legacy_models = [self.model_name, "gemini-1.5-flash", "gemini-1.5-flash-latest", "gemini-pro"]
            
            for m in legacy_models:
                try:
                    model = legacy_genai.GenerativeModel(m)
                    res = model.generate_content(full_prompt)
                    text = self._extract_text(res)
                    if text:
                        return text
                except Exception as inner_e:
                    errors.append(f"legacy({m}): {str(inner_e)}")
                    continue
        except Exception as e2:
            errors.append(f"legacy_init: {str(e2)}")

        if errors:
            return f"AI service error: {errors[-1]}"

        return "AI service temporarily unavailable"

    def generate_general_response(self, prompt: str) -> str:
        return self.generate_response(prompt=prompt, context=None)
