from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from rank_bm25 import BM25Okapi

from app.config import EMBEDDING_MODEL, TOP_K


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


class HybridRetriever:
    def __init__(self, documents: list[Document]):
        self.documents = list(documents)
        self.texts = [doc.page_content for doc in self.documents]
        embeddings = _get_embeddings()
        self.vectorstore = FAISS.from_documents(self.documents, embeddings)
        self.bm25 = BM25Okapi([t.split() for t in self.texts])

    def add_documents(self, new_docs: list[Document]) -> None:
        if not new_docs:
            return
        self.documents.extend(new_docs)
        self.texts.extend([doc.page_content for doc in new_docs])
        new_store = FAISS.from_documents(new_docs, _get_embeddings())
        self.vectorstore.merge_from(new_store)
        self.bm25 = BM25Okapi([t.split() for t in self.texts])

    def retrieve(self, query: str, k: int = None) -> list[Document]:
        k = k or TOP_K
        dense = self.vectorstore.similarity_search(query, k=k)
        dense_texts = {doc.page_content for doc in dense}

        scores = self.bm25.get_scores(query.split())
        top_idx = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        sparse = [
            self.documents[i]
            for i in top_idx
            if self.texts[i] not in dense_texts
        ]

        return (dense + sparse)[: k * 2]
