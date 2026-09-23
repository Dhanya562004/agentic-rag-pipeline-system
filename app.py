"""
Streamlit Web Application for Agentic AI Assistant System
Modern UI with Hybrid RAG + General LLM Fallback
"""

import os
import sys
import time
import streamlit as st

# Ensure 'src' directory is in Python path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if src_path not in sys.path:
    sys.path.append(src_path)

from pipeline import run_pipeline
from retrieval import VectorRetriever, clean_text
from llm_service import LLMService

try:
    from PyPDF2 import PdfReader
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

# Page Configuration
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Vanilla CSS)
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.sub-header {
    font-size: 1.1rem;
    color: #475569;
    margin-bottom: 1.2rem;
}
.metric-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.answer-box {
    background-color: #F0FDF4;
    border: 1px solid #BBF7D0;
    border-radius: 10px;
    padding: 20px;
    font-size: 1.15rem;
    color: #166534;
    font-weight: 500;
    margin-top: 10px;
    margin-bottom: 10px;
}
.explanation-tag {
    font-size: 0.9rem;
    color: #64748B;
    font-weight: 600;
    margin-top: 5px;
}
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# Load Pipeline Services with Streamlit Caching
@st.cache_resource(show_spinner="Initializing AI Pipeline & Vector Index...")
def load_pipeline_services():
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings()
    llm_service = LLMService(provider="gemini")
    return retriever, llm_service


retriever, llm_service = load_pipeline_services()

# Session State for Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Architecture & Document Manager
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("System Architecture")

    st.markdown("""
**Hybrid Routing Logic:**
- **Similarity ≥ 0.75**: 🟢 RAG Mode (Document grounded)
- **Similarity < 0.75**: 🟡 General LLM Fallback (Natural AI response)
""")

    st.divider()

    # Optional Document / PDF File Upload Feature
    st.subheader("📄 Upload Custom Document")
    uploaded_file = st.file_uploader("Upload PDF or TXT to add to RAG Context", type=["pdf", "txt"])

    if uploaded_file is not None:
        try:
            file_text = ""
            if uploaded_file.type == "application/pdf" and PDF_SUPPORT:
                reader = PdfReader(uploaded_file)
                for page in reader.pages:
                    txt = page.extract_text()
                    if txt:
                        file_text += txt + "\n"
            elif uploaded_file.type == "text/plain":
                file_text = uploaded_file.read().decode("utf-8")

            cleaned_file_text = clean_text(file_text)

            if len(cleaned_file_text) >= 50:
                new_chunk = {
                    "chunk_id": f"upload_{int(time.time())}",
                    "text": cleaned_file_text,
                    "source_url": uploaded_file.name,
                    "title": uploaded_file.name,
                    "category": "user_upload",
                    "word_count": len(cleaned_file_text.split()),
                    "date_retrieved": time.strftime("%Y-%m-%d")
                }
                retriever.chunks.append(new_chunk)
                retriever.generate_embeddings(force_regenerate=True)
                st.success(f"Added '{uploaded_file.name}' to RAG context! ({len(cleaned_file_text.split())} words)")
            else:
                st.warning("Uploaded file did not contain enough clean text (min 50 chars).")
        except Exception as upload_err:
            st.error(f"Error processing uploaded document: {upload_err}")

    st.divider()

    model_info = llm_service.get_model_info() if llm_service else {}
    st.caption(f"Model Provider: {model_info.get('provider', 'Gemini').upper()}")
    st.caption(f"Active Chunks: {len(retriever.chunks) if retriever else 0}")
    st.caption("Status: System Ready 🟢")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# Mandatory Guidance Info Banner
st.info("📌 This system answers from documents and falls back to general AI if no match is found.")

# Main Header
st.markdown("<div class='main-header'>🤖 Agentic AI Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>RAG + Smart AI fallback system</div>", unsafe_allow_html=True)

st.divider()

# Sample Query Presets
st.markdown("##### 💡 Try Sample Queries:")
col1, col2, col3 = st.columns(3)

preset_query = ""
if col1.button("📍 Capital of France"):
    preset_query = "What is the capital of France?"
if col2.button("📊 France Economy & GDP"):
    preset_query = "Describe the GDP and economy of France"
if col3.button("💻 Quantum Computing"):
    preset_query = "Explain Quantum Computing"


# User Query Input Box
query_input = st.text_input(
    "Ask anything or upload documents...",
    value=preset_query,
    placeholder="Ask anything or upload documents...",
    key="user_query_input"
)

# Process Query Submission
if st.button("🚀 Submit Query", type="primary") or query_input.strip():

    user_query = query_input.strip()

    if user_query:
        start_time = time.time()
        with st.spinner("Analyzing intent and searching knowledge base..."):
            result = run_pipeline(
                query=user_query,
                retriever=retriever,
                llm_service=llm_service
            )

        elapsed_time = time.time() - start_time
        mode_raw = result.get("mode", "General LLM")
        confidence_raw = result.get("confidence", "Medium")
        score = result.get("score", 0.0)
        answer = result.get("answer", "No response generated.")
        chunks = result.get("retrieved_chunks", [])

        # Format Mode Badge
        if mode_raw == "RAG":
            mode_display = "🟢 RAG (Document Mode)"
            is_rag_mode = True
        elif "Fallback" in mode_raw:
            mode_display = "🟡 RAG → General LLM (Fallback)"
            is_rag_mode = False
        else:
            mode_display = "🔵 General LLM"
            is_rag_mode = False

        # Format Confidence Badge
        if confidence_raw == "High":
            confidence_display = "🟢 High"
        elif confidence_raw == "Medium":
            confidence_display = "🟡 Medium"
        else:
            confidence_display = "🔴 Low"

        # Append to Chat History
        st.session_state.messages.append({
            "query": user_query,
            "answer": answer,
            "mode_display": mode_display,
            "confidence_display": confidence_display,
            "score": score,
            "elapsed_time": elapsed_time,
            "is_rag_mode": is_rag_mode,
            "chunks": chunks if is_rag_mode else []
        })

        st.divider()

        # Display Metrics in 3 Cards in 1 Row
        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric("Mode", mode_display)

        with c2:
            st.metric("Confidence", confidence_display)

        with c3:
            st.metric("Score", f"{score:.2f} ({elapsed_time:.2f}s)")

        # Explanation Label
        if is_rag_mode:
            st.markdown("<div class='explanation-tag'>ℹ️ Generated from document context</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='explanation-tag'>ℹ️ Generated using general AI (no strong document match)</div>", unsafe_allow_html=True)

        # Clean Answer Display Box
        st.subheader("💡 Answer")
        st.success(answer)

        # Context Display (Show ONLY if RAG mode)
        if is_rag_mode and chunks:
            with st.expander("🔍 Retrieved Document Context (Top Chunks)"):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(f"**Source Chunk {i}** (Similarity Score: {chunk.get('similarity_score', 0.0):.2f})")
                    st.write(chunk.get("text", ""))
                    st.caption(f"Source: {chunk.get('source_url', 'Knowledge Base')} | Section: {chunk.get('section', 'Main')}")
                    st.divider()

# Display Session Chat History
if st.session_state.messages:
    st.divider()
    st.subheader("📜 Recent Chat History")
    for idx, msg in enumerate(reversed(st.session_state.messages[:-1])):
        with st.expander(f"Q: {msg['query'][:60]}... ({msg['mode_display']})"):
            st.markdown(f"**Question:** {msg['query']}")
            st.markdown(f"**Answer:** {msg['answer']}")
            st.caption(f"Mode: {msg['mode_display']} | Confidence: {msg['confidence_display']} | Score: {msg['score']:.2f}")