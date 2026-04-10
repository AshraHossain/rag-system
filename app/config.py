import os
from dotenv import load_dotenv

load_dotenv()

# WHY:
# Centralized config → avoids hardcoding secrets
# Fail-fast on missing required values rather than silent errors

# --- LLM Backend ---
# Swap comments to switch between OpenAI and Ollama
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MODEL_NAME = "gpt-4o-mini"

MODEL_NAME = os.getenv("MODEL_NAME", "llama3.2")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- Retrieval Config ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
TOP_K = int(os.getenv("TOP_K", 3))
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# --- Data ---
DATA_PATH = os.getenv("DATA_PATH", "data/sample.txt")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")