from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from app.reranker import rerank
from app.evaluator import evaluate_response
from app.config import MODEL_NAME, OLLAMA_BASE_URL, CHUNK_SIZE, CHUNK_OVERLAP, DATA_PATH

llm = ChatOllama(model=MODEL_NAME, base_url=OLLAMA_BASE_URL)


def load_documents(path=DATA_PATH):
    loader = TextLoader(path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return chunks


def run_rag(query, retriever):
    retrieved_docs = retriever.retrieve(query)
    ranked_docs = rerank(query, retrieved_docs)
    context = "\n".join([doc.page_content for doc in ranked_docs])
    messages = [
        ("system", "Answer ONLY using the provided context."),
        ("user", f"Context:\n{context}\n\nQuestion: {query}")
    ]
    response = llm.invoke(messages)
    answer = response.content
    evaluation = evaluate_response(query, ranked_docs, answer)
    return {
        "answer": answer,
        "evaluation": evaluation
    }