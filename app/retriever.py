from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from app.config import MODEL_NAME, OLLAMA_BASE_URL, TOP_K, EMBEDDING_MODEL


class HybridRetriever:
    def __init__(self, documents):
        self.texts = [doc.page_content for doc in documents]
        self.embeddings = OllamaEmbeddings(
            model=EMBEDDING_MODEL,
            base_url=OLLAMA_BASE_URL
        )
        self.vectorstore = FAISS.from_texts(self.texts, self.embeddings)
        tokenized = [text.split() for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query, k=None):
        k = k or TOP_K
        dense_results = self.vectorstore.similarity_search(query, k=k)
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:k]
        sparse_results = [self.texts[i] for i in top_indices]
        return dense_results + [
            type("Doc", (), {"page_content": text}) for text in sparse_results
        ]