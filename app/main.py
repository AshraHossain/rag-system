import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import CHUNK_OVERLAP, CHUNK_SIZE
from app.evaluator import run_ragas_evaluation
from app.rag_pipeline import load_documents, run_rag
from app.retriever import HybridRetriever

app = FastAPI(title="RAG System API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

docs = load_documents()
retriever = HybridRetriever(docs)


@app.get("/health")
def health():
    return {"status": "ok", "docs_loaded": len(retriever.documents)}


@app.get("/ask")
def ask(query: str):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        return run_rag(query, retriever)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".txt", ".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only .txt and .pdf files are supported",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        if ext == ".pdf":
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(tmp_path)
        else:
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(tmp_path, encoding="utf-8")

        raw = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(raw)
        retriever.add_documents(chunks)
    finally:
        os.unlink(tmp_path)

    return {
        "message": f"Ingested {file.filename}",
        "chunks_added": len(chunks),
        "total_docs": len(retriever.documents),
    }


class EvaluateRequest(BaseModel):
    questions: list[str]
    answers: list[str]
    contexts: list[list[str]]
    ground_truths: list[str] = None


@app.post("/evaluate")
def evaluate(req: EvaluateRequest):
    return run_ragas_evaluation(
        req.questions,
        req.answers,
        req.contexts,
        req.ground_truths,
    )
