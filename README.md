# Production RAG System

Hybrid retrieval · Cross-encoder reranking · Streaming answers · RAGas evaluation

---

## Overview

A production-grade Retrieval-Augmented Generation system that answers questions from your documents. Designed for accuracy over basic RAG by combining keyword and semantic search, re-scoring retrieved passages with a cross-encoder, and streaming LLM responses token-by-token.

**Runs fully locally** — no OpenAI required. Uses Ollama for LLM inference and local HuggingFace models for embeddings and reranking.

---

## Architecture

```
User Query
    ↓
Hybrid Retrieval
  ├── BM25 keyword search (rank-bm25)
  └── Dense vector search (FAISS + bge-small-en-v1.5)
    ↓
Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
    ↓
Context Assembly (top-K passages)
    ↓
LLM Generation — streamed token-by-token (Ollama / OpenRouter)
    ↓
Evaluation (answer relevance · hallucination score · context recall)
    ↓
FastAPI  →  Streamlit UI
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama (local) — mistral:7b, llama3.2, etc. |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (local, no API key) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` (sentence-transformers) |
| Vector store | FAISS (in-memory) |
| Sparse retrieval | BM25 (rank-bm25) |
| RAG framework | LangChain |
| Backend API | FastAPI |
| UI | Streamlit |
| Evaluation | Custom heuristics + RAGas batch evaluation |
| Deployment | Docker Compose |

---

## Features

- **Hybrid retrieval** — BM25 + dense embeddings fused before reranking, improving recall over either alone
- **Cross-encoder reranking** — joint query-document scoring for precision; same technique used by Cohere Rerank
- **Streaming answers** — tokens arrive via SSE as the model generates, no timeout regardless of model speed
- **Document upload** — ingest `.txt` or `.pdf` files at runtime without restarting the server
- **Per-query evaluation** — answer relevance, hallucination score, and context recall shown after every response
- **Batch RAGas evaluation** — `POST /evaluate` endpoint runs faithfulness + answer relevancy via RAGas
- **Dual LLM backend** — switch between local Ollama and OpenRouter with one env var

---

## Project Structure

```
rag-system/
├── app/
│   ├── config.py          # All config from environment variables
│   ├── retriever.py       # HybridRetriever (BM25 + FAISS)
│   ├── reranker.py        # Cross-encoder reranking
│   ├── rag_pipeline.py    # run_rag() and stream_rag() orchestration
│   ├── evaluator.py       # Heuristic metrics + RAGas batch eval
│   └── main.py            # FastAPI app and all endpoints
├── ui/
│   └── streamlit_app.py   # Streamlit frontend
├── data/
│   └── sample.txt         # Sample enterprise knowledge base
├── tests/
│   └── test_evaluator.py  # Unit tests (no LLM/network needed)
├── dockerfile
├── dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Quick Start (Local)

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
- A model pulled: `ollama pull mistral:7b`

### Setup

```bash
git clone https://github.com/AshraHossain/rag-system.git
cd rag-system

python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # macOS/Linux

pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

cp .env.example .env
# Edit .env and set OLLAMA_MODEL to your pulled model name
```

### Run

```bash
# Terminal 1 — API
uvicorn app.main:app --reload

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| API docs (Swagger) | http://localhost:8000/docs |

First startup downloads the embedding model (~130 MB) and reranker (~90 MB) — cached after that.

---

## Docker

```bash
cp .env.example .env   # fill in OLLAMA_MODEL etc.
docker-compose up --build
```

The API container reaches your host Ollama via `host.docker.internal:11434`,
which works automatically on Docker Desktop (Windows and Mac).

**Linux users** — add this to the `api` service in `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```

The HuggingFace embedding and reranker models are stored in a named volume
(`hf_cache`) so they survive `docker-compose down` and only download once.
The UI container is kept minimal — it installs only `streamlit` and `requests`,
not the full ML stack.

---

## Environment Variables

```bash
# LLM backend: "ollama" (default) or "openrouter"
LLM_BACKEND=ollama

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b

# OpenRouter (used when LLM_BACKEND=openrouter)
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=google/gemma-2-9b-it:free

# Embeddings (local, no API key needed)
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Retrieval
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=5
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server status and document count |
| `GET` | `/ask?query=...` | Full response with evaluation metrics |
| `GET` | `/ask/stream?query=...` | Streaming SSE response (token-by-token) |
| `POST` | `/upload` | Ingest a `.txt` or `.pdf` file |
| `POST` | `/evaluate` | Batch RAGas evaluation with ground truths |

---

## Evaluation Metrics

Every `/ask` response includes:

| Metric | Description |
|---|---|
| `answer_relevance` | Word overlap between query and answer |
| `hallucination_score` | Fraction of answer words not found in context (lower = better) |
| `context_recall` | Fraction of answer words present in retrieved context |
| `num_docs_used` | Number of passages passed to the LLM |

For rigorous batch evaluation with ground-truth answers, use `POST /evaluate` which runs RAGas faithfulness and answer relevancy.

---

## Tests

```bash
pytest tests/ -v
```

Unit tests cover all evaluator functions. No LLM or network connection required.

---

## Author

**Ashrafuzzaman M. Hossain** — AI Engineer · LLM Systems · RAG · Agentic AI
