from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_PATH,
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
)
from app.evaluator import evaluate_response
from app.reranker import rerank


def _get_llm():
    if LLM_BACKEND == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            max_retries=5,
        )
    from langchain_ollama import ChatOllama
    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)


def load_documents(path: str = DATA_PATH):
    loader = TextLoader(path, encoding="utf-8")
    raw = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(raw)


def run_rag(query: str, retriever) -> dict:
    llm = _get_llm()
    retrieved = retriever.retrieve(query)
    ranked = rerank(query, retrieved)
    context = "\n\n".join(doc.page_content for doc in ranked)
    messages = [
        (
            "system",
            "Answer ONLY using the provided context. "
            "If the context does not contain enough information, say so.",
        ),
        ("user", f"Context:\n{context}\n\nQuestion: {query}"),
    ]
    answer = llm.invoke(messages).content
    return {
        "answer": answer,
        "sources": [doc.page_content[:300] for doc in ranked],
        "evaluation": evaluate_response(query, ranked, answer),
    }
