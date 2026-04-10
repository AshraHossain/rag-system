# RAG Document Intelligence System

A production-grade **Retrieval-Augmented Generation (RAG)** system built with LangChain, FAISS, and Ollama — enabling context-aware question answering over enterprise documents with hybrid retrieval, reranking, and built-in evaluation metrics.

---

## Architecture

```
User Query
    ↓
FastAPI Service Layer (port 8000)
    ↓
Hybrid Retriever
├── Dense: FAISS + nomic-embed-text (semantic search)
└── Sparse: BM25Okapi (keyword search)
    ↓
CrossEncoder Reranker (ms-marco-MiniLM-L-6-v2)
    ↓
LLM Generation (Ollama / llama3.2)
    ↓
Evaluation Layer (precision@k, hallucination score, answer relevance)
    ↓
Streamlit UI (port 8501)
```

---

## Why Each Component Exists

| Component | Reason |
|---|---|
| **Hybrid Retrieval** | Dense embeddings capture semantic meaning; BM25 handles exact keyword matches. Together they improve recall over either alone. |
| **CrossEncoder Reranker** | Initial retrieval returns noisy results. A cross-encoder jointly scores query-document pairs for significantly higher precision. |
| **Context-Constrained Generation** | Prompting the LLM to answer only from retrieved context reduces hallucination. |
| **Evaluation Layer** | Most RAG systems ship without quality measurement. Built-in metrics enable continuous improvement and production readiness. |
| **FastAPI Layer** | Converts the pipeline into a deployable, scalable REST service. |

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM Backend | Ollama (llama3.2) — runs locally, zero API cost |
| Embeddings | Ollama (nomic-embed-text) |
| Vector Store | FAISS (in-memory, swappable to ChromaDB/Pinecone) |
| Sparse Retrieval | BM25Okapi (rank-bm25) |
| Reranker | CrossEncoder (sentence-transformers, ms-marco-MiniLM-L-6-v2) |
| Document Loading | LangChain TextLoader + RecursiveCharacterTextSplitter |
| UI | Streamlit |
| Config | python-dotenv, fully environment-variable driven |

---

## Project Structure

```
rag-system/
├── app/
│   ├── main.py           # FastAPI service layer
│   ├── rag_pipeline.py   # Orchestrator — ties all components together
│   ├── retriever.py      # Hybrid BM25 + FAISS retriever
│   ├── reranker.py       # CrossEncoder reranker
│   ├── evaluator.py      # precision@k, hallucination score, answer relevance
│   └── config.py         # Centralized config via environment variables
├── data/
│   └── sample.txt        # Document corpus (replace with your documents)
├── ui/
│   └── streamlit_app.py  # Chat interface
├── tests/
├── .env                  # Local config (not committed)
├── requirements.txt
├── dockerfile
└── README.md
```

---

## Quickstart

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running

### 1. Clone and set up environment

```bash
git clone https://github.com/AshraHossain/rag-system.git
cd rag-system
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Pull required Ollama models

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if needed — defaults work out of the box
```

### 4. Add your documents

```bash
# Replace or add documents in the data/ folder
# Default: data/sample.txt
```

### 5. Start the system (3 terminals)

```bash
# Terminal 1 — Ollama (if not already running)
ollama serve

# Terminal 2 — FastAPI backend
uvicorn app.main:app --reload --port 8000

# Terminal 3 — Streamlit UI
streamlit run ui/streamlit_app.py
```

### 6. Open the UI

```
http://localhost:8501
```

Or query the API directly:
```
http://localhost:8000/ask?query=What+is+RAG
http://localhost:8000/docs   ← Swagger UI
```

---

## API

### `GET /ask`

| Parameter | Type | Description |
|---|---|---|
| `query` | string | Question to ask against the document corpus |

**Response:**
```json
{
  "answer": "RAG stands for Retrieval Augmented Generation...",
  "evaluation": {
    "precision_at_k": 0.5,
    "hallucination_score": 0.646,
    "answer_relevance": 0.462,
    "num_docs_used": 2
  }
}
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| `precision_at_k` | Ratio of retrieved docs that match relevant set |
| `hallucination_score` | Inverse context overlap — lower means more grounded |
| `answer_relevance` | Query-answer topic overlap |
| `num_docs_used` | Number of docs passed to LLM after reranking |

> Note: Ground truth is currently simulated. In production, replace with labeled evaluation datasets (e.g., RAGAS framework).

---

## Configuration

All parameters are environment-variable driven via `.env`:

```env
MODEL_NAME=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=3
DATA_PATH=data/sample.txt
```

---

## Deployment

### Option 1 — HuggingFace Spaces (Demo)
Fastest path to a public demo for portfolio/recruiter access.

### Option 2 — AWS (Production)
- FastAPI → ECS / EC2
- Vector DB → Pinecone
- Document storage → S3
- API Gateway for routing

### Option 3 — Docker
```bash
docker build -t rag-system .
docker run -p 8000:8000 rag-system
```

---

## Roadmap

- [ ] Multi-document upload via Streamlit UI
- [ ] Persistent vector store (ChromaDB)
- [ ] Cross-encoder reranking with Cohere API
- [ ] RAGAS evaluation integration
- [ ] Observability dashboard (response time, token usage, retrieval accuracy)
- [ ] Docker Compose (API + UI as services)
- [ ] Query caching layer

---

## Real-World Use Cases

- **Enterprise Knowledge Assistant** — internal documentation search
- **Developer / SDET Assistant** — debug failures using logs and API docs
- **Compliance Systems** — FAA / FDA / legal documentation retrieval with traceability
- **Customer Support Automation** — FAQ answering from knowledge base

---

## Author

**Ashraf Hossain** — IBM-Certified Generative AI & Agentic AI Engineer  
16+ years in safety-critical software engineering (FAA aviation systems)

[GitHub](https://github.com/AshraHossain) · [LinkedIn](https://linkedin.com/in/ashrafmhossain)