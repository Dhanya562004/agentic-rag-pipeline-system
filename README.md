# Agentic RAG Pipeline System 🚀
> *A modular, production-ready AI pipeline that intelligently routes, retrieves, generates, evaluates, and logs user queries.*

---

## 📌 Overview

The **Agentic RAG Pipeline System** is an AI application built with Python that elevates a standard Retrieval-Augmented Generation (RAG) pipeline into a clean, modular, production-style system.

Instead of running document retrieval for every query, the system uses an intelligent **Agent Router** to decide whether a user's question requires domain document context (RAG mode) or general intelligence (General LLM mode). The response is then automatically evaluated for confidence and logged for performance tracking.

---

## ✨ Key Features

- **🧠 Agent-Based Routing**: Automatically routes queries to RAG mode (document retrieval) or General LLM mode based on keyword intent.
- **⚡ End-to-End Orchestration**: Seamlessly coordinates routing &rarr; retrieval &rarr; LLM generation &rarr; evaluation &rarr; logging.
- **📊 Real-Time Evaluation**: Computes confidence ratings (`High`, `Medium`, `Low`) and checks answer relevance without external model overhead.
- **🔍 Built-in Observability**: Logs latency, execution mode, response quality, and error diagnostics to standard output.
- **🎨 Clean Streamlit UI**: User-friendly web application for testing queries without needing external API tools.
- **🧱 Modular Architecture**: Lightweight, beginner-friendly code structure designed for high readability and easy interview walkthroughs.

---

## 🏗️ Architecture

The application follows a clean 5-step lifecycle for every incoming query:

```
[User Query] 
     │
     ▼
[1. Agent Router] ──────► Decides mode: 'RAG' vs 'General'
     │
     ▼
[2. Retrieval Layer] ───► (If RAG) Fetches top relevant document chunks using vector embeddings
     │
     ▼
[3. LLM Service] ───────► Generates answer using context or general knowledge
     │
     ▼
[4. Evaluation Layer] ──► Assesses response relevance & assigns confidence score (High/Medium/Low)
     │
     ▼
[5. Observability Logger] ──► Logs response time, query details, and status
```

---

## 🛠️ Tech Stack

- **Core Language**: Python 3.10+
- **User Interface**: Streamlit
- **API Server**: FastAPI + Uvicorn
- **Embeddings & Vector Search**: Sentence Transformers (`m2-bert-80M-8k-retrieval`), NumPy, Scikit-Learn
- **LLM Integration**: Together AI (Meta LLaMA 3.1 8B Instruct) / OpenAI fallback

---

## 📁 Project Structure

```
Agentic RAG Pipeline System/
├── app.py                  # Root Streamlit Web Application
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
├── data/
│   ├── processed_data.json # Chunked knowledge base data
│   └── embeddings.pkl      # Precomputed vector embeddings
├── src/
│   ├── agent.py            # Simple rule-based query router
│   ├── evaluator.py        # Rule-based answer confidence & relevance scorer
│   ├── logger.py           # Latency and observability logging module
│   ├── pipeline.py         # Main pipeline orchestrator (run_pipeline)
│   ├── retrieval.py        # Vector embedding generation & hybrid document retrieval
│   ├── llm_service.py      # LLM response generator (Together AI / OpenAI)
│   ├── main.py             # FastAPI REST endpoints (/ask, /retrieve, /generate)
│   └── start_services.py   # Launcher script with service health checks
└── static/                 # Static web interface pages
```

### File Breakdown:
- **`app.py`**: Streamlit interface that directly calls the pipeline to deliver a clean interactive UI.
- **`src/pipeline.py`**: Main entry function `run_pipeline(query)` that coordinates all pipeline steps.
- **`src/agent.py`**: Inspects user queries to route them to RAG or General mode.
- **`src/evaluator.py`**: Evaluates response quality and output confidence score.
- **`src/logger.py`**: Tracks latency, query statistics, and runtime errors.
- **`src/retrieval.py`**: Generates embeddings and performs hybrid document retrieval.
- **`src/llm_service.py`**: Interacts with Together AI or OpenAI APIs to generate concise answers.

---

## 🔑 API Key Setup

The system uses **Together AI** (or optionally OpenAI) for text generation.

Set your API key as an environment variable:

### Windows (PowerShell):
```powershell
$env:TOGETHER_API_KEY="your_together_api_key_here"
```

### Linux / macOS:
```bash
export TOGETHER_API_KEY="your_together_api_key_here"
```

---

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Dhanya562004/agentic-rag-pipeline-system.git
cd agentic-rag-pipeline-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 🌐 Deployment (Streamlit Cloud)

To deploy this project live on **Streamlit Cloud**:

1. Push your code to your GitHub repository.
2. Sign in to [Streamlit Cloud](https://share.streamlit.io/).
3. Click **New app** and select your repository (`Dhanya562004/agentic-rag-pipeline-system`, Branch: `main`).
4. Set **Main file path** to `app.py`.
5. Under **Advanced Settings &rarr; Secrets**, add your API key:
   ```toml
   TOGETHER_API_KEY = "your_actual_api_key"
   ```
6. Click **Deploy!**

---

## 📝 Example Output

When a user submits a query, the system returns a structured output:

### Example 1: RAG Query (Document-based)
- **Input Query**: *"What is the capital of France?"*
- **JSON Result**:
  ```json
  {
    "mode": "rag",
    "answer": "Paris is the capital of France.",
    "confidence": "High"
  }
  ```

### Example 2: General Query
- **Input Query**: *"What is quantum computing?"*
- **JSON Result**:
  ```json
  {
    "mode": "general",
    "answer": "Quantum computing is a field of computing focused on developing technology based on the principles of quantum theory.",
    "confidence": "High"
  }
  ```

---

## 🔮 Future Improvements

- [ ] **Multi-Agent Collaboration**: Add specialized agent roles (e.g. Fact-Checker Agent, Summarizer Agent).
- [ ] **Dynamic Knowledge Base Ingestion**: Allow users to upload custom PDF/TXT documents directly via the UI.
- [ ] **Advanced Reranking**: Integrate cross-encoder reranking models to boost document retrieval precision.
- [ ] **Persistent Log Database**: Store observability logs in SQLite/PostgreSQL for long-term analytics dashboards.

---

## 🤝 Contributing & License

Contributions are welcome! Feel free to open an issue or submit a pull request.
Distributed under the MIT License.
