import time
import streamlit as st
from retrieval import VectorRetriever, extract_text_from_file
from pipeline import run_pipeline
from llm_service import LLMService

# ------------------------------------
# 1. PAGE CONFIG & DARK MODERN THEME
# ------------------------------------
st.set_page_config(
    page_title="Agentic AI Assistant",
    page_icon="🤖",
    layout="wide"
)

st.markdown("""
<style>
    /* Dark Modern Base Styles */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Styling */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #58a6ff;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.1rem;
        color: #8b949e;
        margin-bottom: 30px;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
    }

    /* Answer Container */
    .answer-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-left: 4px solid #58a6ff;
        border-radius: 8px;
        padding: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
        color: #e6edf3;
        margin-top: 15px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------
# 2. INITIALIZE SERVICES & STATE
# ------------------------------------
@st.cache_resource
def get_llm_service():
    return LLMService()

llm_service = get_llm_service()

if "retriever" not in st.session_state:
    st.session_state.retriever = VectorRetriever()

retriever = st.session_state.retriever

# ------------------------------------
# 3. SIDEBAR (FILE UPLOAD & CHUNKS)
# ------------------------------------
with st.sidebar:
    st.title("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file is not None:
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_uploaded_file") != file_key:
            with st.spinner("Extracting text and generating embeddings..."):
                raw_text = extract_text_from_file(uploaded_file)
                if raw_text.strip():
                    num_chunks = retriever.add_document(raw_text)
                    st.session_state.last_uploaded_file = file_key
                    st.success(f"Successfully processed {num_chunks} chunks!")
                else:
                    st.error("Could not extract readable text from the document.")

    st.markdown("---")
    st.metric(label="Total Document Chunks", value=len(retriever.chunks))

# ------------------------------------
# 4. MAIN UI CONTENT
# ------------------------------------
st.markdown('<h1 class="main-title">Agentic AI Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG + Smart AI Fallback System</p>', unsafe_allow_html=True)

query_input = st.text_input("Ask your question...", placeholder="Type your query here...")
ask_button = st.button("Ask", type="primary")

if ask_button and query_input:
    start_time = time.time()
    
    with st.spinner("Thinking..."):
        result = run_pipeline(
            query=query_input,
            retriever=retriever,
            llm_service=llm_service
        )
    
    elapsed_time = time.time() - start_time

    mode = result.get("mode", "🔵 LLM")
    score = result.get("score", 0.0)
    answer = result.get("answer", "")
    retrieved_chunks = result.get("retrieved_chunks", [])

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Mode</div>
            <div class="metric-value">{mode}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Similarity Score</div>
            <div class="metric-value">{score:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Response Time</div>
            <div class="metric-value">{elapsed_time:.2f}s</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Answer")
    st.markdown(f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True)

    if mode == "🟢 RAG" and retrieved_chunks:
        with st.expander("🔍 Retrieved Context"):
            for i, chunk in enumerate(retrieved_chunks, 1):
                st.markdown(f"**Chunk {i} (Score: {chunk['score']:.2f})**")
                st.write(chunk["text"])
                st.markdown("---")
