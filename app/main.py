from fastapi import FastAPI
from app.rag_pipeline import load_documents, run_rag
from app.retriever import HybridRetriever

app = FastAPI()

docs = load_documents()
retriever = HybridRetriever(docs)


@app.get("/ask")
def ask(query: str):
    result = run_rag(query, retriever)
    return result