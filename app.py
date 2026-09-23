"""
Streamlit Web Application for Agentic RAG Pipeline System
"""

import os
import sys
import streamlit as st

# Ensure 'src' directory is in Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pipeline import run_pipeline
from retrieval import VectorRetriever
from llm_service import LLMService

# Page config
st.set_page_config(
    page_title="Agentic RAG Pipeline System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ FIXED CSS (IMPORTANT)
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1E293B;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1.05rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}
.metric-card {
    background-color: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
.stAlert {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# Load services
@st.cache_resource(show_spinner="Initializing AI Pipeline...")
def load_pipeline_services():
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings()

    llm_service = LLMService(provider="gemini")

    return retriever, llm_service


retriever, llm_service = load_pipeline_services()


# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("System Architecture")

    st.markdown("""
**Agentic Workflow:**

1. Intent Router → RAG or General LLM  
2. Vector Retrieval (Embeddings)  
3. Confidence Check  
4. Auto Fallback to LLM  
""")

    st.divider()

    model_info = llm_service.get_model_info() if llm_service else {}

    st.caption(f"Model: {model_info.get('model', 'Default')}")
    st.caption(f"Chunks: {len(retriever.chunks) if retriever else 0}")
    st.caption("Status: Ready 🟢")


# Header
st.markdown("<div class='main-header'>🤖 Agentic RAG Pipeline System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>RAG + Smart LLM fallback system</div>", unsafe_allow_html=True)

st.divider()


# Sample queries
st.markdown("### 💡 Try Sample Queries:")

col1, col2, col3 = st.columns(3)

preset_query = ""

if col1.button("📍 Capital of France"):
    preset_query = "What is the capital of France?"

if col2.button("📊 France economy"):
    preset_query = "Describe the GDP and economy of France"

if col3.button("💻 Quantum Computing"):
    preset_query = "Explain Quantum Computing"


# Input
query_input = st.text_input(
    "Enter your question:",
    value=preset_query,
    placeholder="Ask anything...",
)


# Button
if st.button("🚀 Submit Query"):

    if not query_input.strip():
        st.warning("Please enter a question")
    else:
        with st.spinner("Processing..."):

            result = run_pipeline(
                query=query_input.strip(),
                retriever=retriever,
                llm_service=llm_service
            )

            mode = result.get("mode", "N/A")
            confidence = result.get("confidence", "Medium")
            score = result.get("score", 0.0)
            answer = result.get("answer", "No response")
            chunks = result.get("retrieved_chunks", [])

            st.divider()

            # Metrics
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Mode", mode)

            with c2:
                st.metric("Confidence", confidence)

            with c3:
                st.metric("Score", f"{score:.2f}")

            # Answer
            st.subheader("💡 Answer")
            st.success(answer)

            # Sources
            if chunks:
                with st.expander("🔍 Retrieved Context"):
                    for i, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Source {i}**")
                        st.write(chunk.get("text", ""))
                        st.divider()