import os
from dotenv import load_dotenv

load_dotenv(override=True)

# LLM backend: "ollama" or "openrouter"
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")

# Ollama (local)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3")

# OpenRouter (fallback / cloud)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Embeddings — local sentence-transformers (no API key needed)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 5))

# Reranking
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Data
DATA_PATH = os.getenv("DATA_PATH", "data/sample.txt")
