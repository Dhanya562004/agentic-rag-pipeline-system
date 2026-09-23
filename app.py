"""
🔥 Agentic AI Assistant - Modern UI
RAG + General LLM Hybrid System
"""

import os
import sys
import time
import streamlit as st

# Path setup
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from pipeline import run_pipeline
from retrieval import VectorRetriever, clean_text
from llm_service import LLMService

try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except:
    PDF_SUPPORT = False

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Agentic AI", page_icon="🤖", layout="wide")

# -----------------------
# 🎨 MODERN CSS
# -----------------------
st.markdown("""
<style>

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

/* Header */
.title {
    font-size: 2.8rem;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}

/* Cards */
.card {
    background: rgba(255,255,255,0.05);
    padding: 18px;
    border-radius: 15px;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Answer Box */
.answer {
    background: rgba(16,185,129,0.1);
    padding: 20px;
    border-radius: 12px;
    font-size: 1.2rem;
    margin-top: 10px;
}

/* Input */
input {
    border-radius: 10px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#06b6d4);
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}

/* Mode Colors */
.rag {color:#22c55e;font-weight:bold;}
.fallback {color:#facc15;font-weight:bold;}
.llm {color:#38bdf8;font-weight:bold;}

</style>
""", unsafe_allow_html=True)


# -----------------------
# LOAD SERVICES
# -----------------------
@st.cache_resource
def load_services():
    r = VectorRetriever()
    r.load_chunks()
    r.generate_embeddings()
    l = LLMService(provider="gemini")
    return r, l

retriever, llm_service = load_services()


# -----------------------
# CHUNK HELPER
# -----------------------
def split_chunks(text, size=300):
    words = text.split()
    return [
        " ".join(words[i:i+size])
        for i in range(0, len(words), size)
        if len(" ".join(words[i:i+size])) > 50
    ]


# -----------------------
# 📄 SIDEBAR UPLOAD
# -----------------------
with st.sidebar:
    st.title("📄 Upload Document")

    file = st.file_uploader("Upload PDF / TXT", type=["pdf","txt"])

    if file:
        try:
            text = ""

            if file.type == "application/pdf" and PDF_SUPPORT:
                reader = PdfReader(file)
                for p in reader.pages:
                    t = p.extract_text()
                    if t:
                        text += t

            elif file.type == "text/plain":
                text = file.read().decode("utf-8")

            cleaned = clean_text(text)
            chunks = split_chunks(cleaned)

            count = 0
            for c in chunks:
                retriever.chunks.append({
                    "chunk_id": f"upload_{time.time()}_{count}",
                    "text": c,
                    "source_url": file.name
                })
                count += 1

            if count > 0:
                retriever.generate_embeddings(force_regenerate=True)
                st.success(f"✅ {count} chunks added!")
            else:
                st.warning("No useful content found")

        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.caption(f"Chunks: {len(retriever.chunks)}")


# -----------------------
# HEADER
# -----------------------
st.markdown('<div class="title">🤖 Agentic AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">RAG + Smart AI Fallback System</div>', unsafe_allow_html=True)

st.info("📌 Upload documents OR ask anything — system will decide automatically.")

# -----------------------
# INPUT
# -----------------------
query = st.text_input("Ask your question...")

if st.button("🚀 Ask") and query:

    start = time.time()

    result = run_pipeline(query, retriever, llm_service)

    elapsed = time.time() - start

    mode = result.get("mode")
    score = result.get("score",0)
    answer = result.get("answer","")
    chunks = result.get("retrieved_chunks",[])

    # -----------------------
    # MODE UI
    # -----------------------
    if mode == "RAG":
        mode_label = '<span class="rag">🟢 RAG (Document)</span>'
        explanation = "Answer from document context"
        show_context = True
    elif "Fallback" in mode:
        mode_label = '<span class="fallback">🟡 RAG → LLM (Fallback)</span>'
        explanation = "No strong match → General AI used"
        show_context = False
    else:
        mode_label = '<span class="llm">🔵 General LLM</span>'
        explanation = "Pure AI response"
        show_context = False

    # -----------------------
    # METRICS
    # -----------------------
    c1,c2,c3 = st.columns(3)

    with c1:
        st.markdown(f"<div class='card'>Mode<br>{mode_label}</div>", unsafe_allow_html=True)

    with c2:
        st.markdown(f"<div class='card'>Score<br>{score:.2f}</div>", unsafe_allow_html=True)

    with c3:
        st.markdown(f"<div class='card'>Time<br>{elapsed:.2f}s</div>", unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:10px;color:#94a3b8'>{explanation}</div>", unsafe_allow_html=True)

    # -----------------------
    # ANSWER
    # -----------------------
    st.subheader("💡 Answer")
    st.markdown(f"<div class='answer'>{answer}</div>", unsafe_allow_html=True)

    # -----------------------
    # CONTEXT
    # -----------------------
    if show_context and chunks:
        with st.expander("🔍 Retrieved Context"):
            for c in chunks:
                st.write(c["text"])