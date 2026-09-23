"""
Streamlit Web Application for Agentic RAG Pipeline System
Production-ready UI displaying Execution Mode, Confidence Score, Answer, and Source Chunks.
Runs with: streamlit run app.py
"""

import os
import sys
import streamlit as st

# Ensure 'src' directory is in Python path for clean module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pipeline import run_pipeline
from retrieval import VectorRetriever
from llm_service import LLMService

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Pipeline System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern UI aesthetics
st.markdown("""
<style:
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
</style:
""", unsafe_allow_html=True)

# Cache pipeline services to prevent reloading embeddings on every interaction
@st.cache_resource(show_spinner="Initializing Embeddings & AI Pipeline Services...")
def load_pipeline_services():
    """
    Load vector retriever and LLM service efficiently with caching.
    """
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings()
    
    # Initialize LLM Service (handles missing keys gracefully with fallback)
    llm_service = LLMService(provider="gemini")
        
    return retriever, llm_service

# Load cached services
retriever, llm_service = load_pipeline_services()

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=64)
    st.title("System Architecture")
    st.markdown("""
    **Agentic Workflow:**
    1. **Intent Router:** Classifies query into *RAG (Document)* or *General Knowledge (LLM)*.
    2. **RAG Retrieval:** Vector similarity search using sentence embeddings.
    3. **Confidence Evaluator:** Threshold check (`similarity score < 0.5`).
    4. **Intelligent Fallback:** Automatically invokes General LLM if RAG confidence is low or no document is found.
    """)
    st.divider()
    model_info = llm_service.get_model_info() if llm_service else {}
    st.caption(f"**LLM Model:** {model_info.get('model', 'Default')}")
    st.caption(f"**Retriever Chunks:** {len(retriever.chunks) if retriever else 0}")
    st.caption("**Status:** System Ready 🟢")

# Main Page Header
st.markdown("<div class='main-header'>🤖 Agentic RAG Pipeline System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Intelligent decision-making pipeline with RAG document retrieval and automatic General LLM fallback.</div>", unsafe_allow_html=True)

st.divider()

# Sample Query Selector
st.markdown("##### 💡 Try Sample Queries:")
col_sample1, col_sample2, col_sample3 = st.columns(3)

preset_query = ""
if col_sample1.button("📍 What is the capital of France?"):
    preset_query = "What is the capital of France?"
if col_sample2.button("📊 Describe the GDP and economy of France"):
    preset_query = "Describe the GDP and economy of France"
if col_sample3.button("💻 Explain Quantum Computing"):
    preset_query = "Explain Quantum Computing"

# User Input Field
query_input = st.text_input(
    label="Enter your question:", 
    value=preset_query if preset_query else "",
    placeholder="e.g., What is the climate of France or How does a neural network work?",
    key="user_query"
)

# Execution Button
if st.button("Submit Query", type="primary", use_container_width=False):
    if not query_input or not query_input.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("🤖 Agent analyzing query, routing intent, and processing answer..."):
            result = run_pipeline(
                query=query_input.strip(),
                retriever=retriever,
                llm_service=llm_service
            )
            
            mode = result.get("mode", "N/A")
            confidence = result.get("confidence", "Medium")
            score = result.get("score", 0.0)
            answer = result.get("answer", "No response generated.")
            chunks = result.get("retrieved_chunks", [])
            
            st.divider()
            
            # Display Key Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            
            # Metric 1: Execution Mode with visual badges
            with m_col1:
                if "Fallback" in mode:
                    st.metric(label="Execution Mode", value="RAG → LLM (Fallback)", delta="Automatic Fallback", delta_color="off")
                elif mode.upper() == "RAG":
                    st.metric(label="Execution Mode", value="RAG (Document Mode)", delta="Document Grounded", delta_color="normal")
                else:
                    st.metric(label="Execution Mode", value="General LLM", delta="General Knowledge", delta_color="off")

            # Metric 2: Confidence Score
            with m_col2:
                if confidence == "High":
                    st.metric(label="Confidence Rating", value="🟢 High Confidence")
                elif confidence == "Medium":
                    st.metric(label="Confidence Rating", value="🟡 Medium Confidence")
                else:
                    st.metric(label="Confidence Rating", value="🔴 Low Confidence")

            # Metric 3: Numerical Score
            with m_col3:
                st.metric(label="Similarity / Rating Score", value=f"{score:.2f}")

            # Display Final Answer Box
            st.subheader("💡 Final Answer")
            st.info(answer)

            # Display Source Chunks if RAG was used
            if chunks:
                with st.expander(f"🔍 View Retrieved Context ({len(chunks)} Chunks)"):
                    for idx, chunk in enumerate(chunks, 1):
                        st.markdown(f"**Source {idx}: {chunk.get('title', 'Document')}** (Similarity Score: `{chunk.get('similarity_score', 0.0):.3f}`)")
                        st.write(chunk.get("text", ""))
                        st.caption(f"Category: {chunk.get('category', 'N/A')} | Section: {chunk.get('section', 'N/A')}")
                        st.divider()

