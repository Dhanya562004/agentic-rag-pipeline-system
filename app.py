"""
Streamlit UI for Agentic RAG Pipeline System
Allows testing the RAG + General LLM Pipeline directly without external API calls.
"""

import os
import sys
import streamlit as st

# Add 'src' directory to Python path for clean module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from pipeline import run_pipeline
from retrieval import VectorRetriever
from llm_service import LLMService

# Page configuration
st.set_page_config(
    page_title="Agentic RAG Pipeline System",
    page_icon="🤖",
    layout="centered"
)

# Cache pipeline services to prevent reloading embeddings on every interaction
@st.cache_resource
def load_pipeline_services():
    # Initialize Retriever (loads chunks and embeddings)
    retriever = VectorRetriever()
    retriever.load_chunks()
    retriever.generate_embeddings()
    
    # Initialize LLM Service (graceful fallback if API key is not configured)
    try:
        llm_service = LLMService(provider="together")
    except Exception as e:
        st.warning(f"LLM Service notice: {e}. Falling back to default responses.")
        llm_service = None
        
    return retriever, llm_service

# Title and Description
st.title("🤖 Agentic RAG Pipeline System")
st.write("Enter your question below. The agent automatically decides whether to route it to **RAG (Document Context)** or **General LLM**.")

# Initialize cached services
retriever, llm_service = load_pipeline_services()

# Input box for user query
query = st.text_input("Enter your question:", placeholder="e.g. What is the capital of France?")

# Ask button
if st.button("Ask", type="primary"):
    if not query.strip():
        st.error("Please enter a question before submitting.")
    else:
        with st.spinner("Running Agentic Pipeline..."):
            result = run_pipeline(
                query=query,
                retriever=retriever,
                llm_service=llm_service
            )
            
            st.divider()
            
            # Display mode and confidence
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Execution Mode", value=result.get("mode", "N/A").upper())
            with col2:
                st.metric(label="Confidence Rating", value=result.get("confidence", "N/A"))
                
            # Display generated answer
            st.subheader("Answer")
            st.write(result.get("answer", "No response generated."))
