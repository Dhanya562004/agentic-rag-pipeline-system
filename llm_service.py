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

        # 1. Primary: Try modern google-genai SDK with dynamic model discovery
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            
            available_models = []
            try:
                for m in client.models.list():
                    name = getattr(m, "name", "") or str(m)
                    clean_name = name.replace("models/", "")
                    if "gemini" in clean_name and "embed" not in clean_name and "imagen" not in clean_name:
                        available_models.append(clean_name)
            except Exception as list_err:
                errors.append(f"genai_list: {str(list_err)}")

            candidates = [self.model_name, "gemini-1.5-flash", "gemini-2.0-flash", "gemini-2.5-flash", "gemini-1.5-pro"]
            for m in available_models:
                if m not in candidates:
                    candidates.append(m)

            for m in candidates:
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
                    if "api_key" in err_text or "unauthenticated" in err_text or "invalid" in err_text or "blocked" in err_text:
                        return f"API Key Error: {err_msg}"
                    break
        except Exception as e1:
            errors.append(f"genai_init: {str(e1)}")

        # 2. Backup: Try legacy google-generativeai SDK with dynamic model discovery
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            
            legacy_available = []
            try:
                for m in legacy_genai.list_models():
                    if hasattr(m, "supported_generation_methods") and "generateContent" in m.supported_generation_methods:
                        c_name = m.name.replace("models/", "")
                        if "embed" not in c_name and "imagen" not in c_name and "pro-001" not in c_name:
                            legacy_available.append(c_name)
            except Exception as legacy_list_err:
                errors.append(f"legacy_list: {str(legacy_list_err)}")

            candidates_legacy = [self.model_name, "gemini-1.5-flash", "gemini-1.5-flash-latest"]
            for m in legacy_available:
                if m not in candidates_legacy:
                    candidates_legacy.append(m)

            for m in candidates_legacy:
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
            for err in reversed(errors):
                if "api_key" in err.lower() or "invalid" in err.lower() or "unauthenticated" in err.lower():
                    return f"API Key Error: {err}"
            return f"AI service error: {errors[-1]}"

        return "AI service temporarily unavailable"

    def generate_general_response(self, prompt: str) -> str:
        return self.generate_response(prompt=prompt, context=None)
