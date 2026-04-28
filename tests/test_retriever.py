"""
20 tests for HybridRetriever covering BM25 + dense retrieval on both
structured (tabular / key-value) and unstructured (prose) documents.

All tests use a deterministic hash-based embedding so no model download
or network connection is needed.
"""
from unittest.mock import patch

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.retriever import HybridRetriever


# ---------------------------------------------------------------------------
# Deterministic fake embeddings (no model required)
# ---------------------------------------------------------------------------

class _HashEmbeddings(Embeddings):
    """Embeds text by hashing each word into a fixed-dim vector.

    Documents with overlapping vocabulary get similar vectors, so the
    FAISS similarity search behaves sensibly for keyword-overlap queries.
    """

    DIM = 128

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        for word in text.lower().split():
            vec[hash(word) % self.DIM] += 1.0
        norm = sum(v ** 2 for v in vec) ** 0.5
        return [v / norm for v in vec] if norm else vec


_FAKE_EMB = _HashEmbeddings()
_PATCH = patch("app.retriever._get_embeddings", return_value=_FAKE_EMB)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _doc(text: str, source: str = "test") -> Document:
    return Document(page_content=text, metadata={"source": source})


UNSTRUCTURED = [
    _doc("Retrieval augmented generation combines search with LLM generation"),
    _doc("Hybrid retrieval uses BM25 keyword search and dense vector embeddings"),
    _doc("Cross-encoder reranking improves precision by scoring query-document pairs"),
    _doc("FAISS is a library for efficient similarity search of dense vectors"),
    _doc("Python is a general purpose programming language for data science"),
    _doc("LangChain provides abstractions for building RAG pipelines quickly"),
    _doc("Transformers use self-attention to model long-range dependencies in text"),
]

STRUCTURED = [
    _doc("Name: John Smith | Role: Software Engineer | Department: Engineering"),
    _doc("Product: MeridianFlow | Price: $5000/month | Category: Data Pipeline"),
    _doc("Invoice: INV-2024-001 | Amount: $12450 | Status: Paid | Date: 2024-01-15"),
    _doc("Server: prod-api-01 | CPU: 85% | Memory: 12GB/16GB | Status: Warning"),
    _doc("Employee: Jane Doe | Salary: $120000 | Department: Sales | Joined: 2022"),
    _doc("Ticket: TKT-999 | Priority: P1 | Status: Open | Assignee: Alice"),
    _doc("Config: max_retries=5 | timeout=30 | log_level=INFO | region=us-east-1"),
]

MIXED = UNSTRUCTURED + STRUCTURED


@pytest.fixture
def unstructured_retriever():
    with _PATCH:
        return HybridRetriever(list(UNSTRUCTURED))


@pytest.fixture
def structured_retriever():
    with _PATCH:
        return HybridRetriever(list(STRUCTURED))


@pytest.fixture
def mixed_retriever():
    with _PATCH:
        return HybridRetriever(list(MIXED))


# ---------------------------------------------------------------------------
# 1–3  Initialisation
# ---------------------------------------------------------------------------

def test_init_stores_all_documents():
    with _PATCH:
        r = HybridRetriever(list(UNSTRUCTURED))
    assert len(r.documents) == len(UNSTRUCTURED)


def test_init_bm25_corpus_size_matches():
    with _PATCH:
        r = HybridRetriever(list(STRUCTURED))
    assert r.bm25.corpus_size == len(STRUCTURED)


def test_init_vectorstore_is_created():
    with _PATCH:
        r = HybridRetriever(list(UNSTRUCTURED[:3]))
    assert r.vectorstore is not None


# ---------------------------------------------------------------------------
# 4–7  Basic retrieve behaviour
# ---------------------------------------------------------------------------

def test_retrieve_returns_list_of_documents(unstructured_retriever):
    results = unstructured_retriever.retrieve("retrieval augmented generation")
    assert isinstance(results, list)
    assert all(isinstance(d, Document) for d in results)


def test_retrieve_unstructured_keyword_match(unstructured_retriever):
    results = unstructured_retriever.retrieve("BM25 keyword search embeddings")
    texts = [d.page_content for d in results]
    assert any("BM25" in t for t in texts)


def test_retrieve_respects_max_result_count(unstructured_retriever):
    k = 3
    results = unstructured_retriever.retrieve("search retrieval", k=k)
    assert len(results) <= k * 2


def test_retrieve_custom_k(unstructured_retriever):
    results_k2 = unstructured_retriever.retrieve("search", k=2)
    results_k4 = unstructured_retriever.retrieve("search", k=4)
    assert len(results_k2) <= len(results_k4)


# ---------------------------------------------------------------------------
# 8–10  No duplicates
# ---------------------------------------------------------------------------

def test_retrieve_no_duplicate_unstructured(unstructured_retriever):
    results = unstructured_retriever.retrieve("retrieval search embeddings")
    texts = [d.page_content for d in results]
    assert len(texts) == len(set(texts))


def test_retrieve_no_duplicate_structured(structured_retriever):
    results = structured_retriever.retrieve("Department Engineering Sales")
    texts = [d.page_content for d in results]
    assert len(texts) == len(set(texts))


def test_retrieve_no_duplicate_mixed_corpus(mixed_retriever):
    results = mixed_retriever.retrieve("retrieval BM25 search")
    texts = [d.page_content for d in results]
    assert len(texts) == len(set(texts))


# ---------------------------------------------------------------------------
# 11–13  Structured data queries
# ---------------------------------------------------------------------------

def test_retrieve_structured_by_field_name(structured_retriever):
    results = structured_retriever.retrieve("Invoice Amount Status")
    texts = [d.page_content for d in results]
    assert any("Invoice" in t for t in texts)


def test_retrieve_structured_by_value(structured_retriever):
    results = structured_retriever.retrieve("Engineering Department")
    texts = [d.page_content for d in results]
    assert any("Engineering" in t for t in texts)


def test_retrieve_structured_numeric_field(structured_retriever):
    results = structured_retriever.retrieve("CPU Memory Server Status")
    texts = [d.page_content for d in results]
    assert any("CPU" in t or "Memory" in t for t in texts)


# ---------------------------------------------------------------------------
# 14–16  Mixed structured + unstructured corpus
# ---------------------------------------------------------------------------

def test_retrieve_mixed_corpus_returns_results(mixed_retriever):
    results = mixed_retriever.retrieve("retrieval generation embeddings")
    assert len(results) > 0


def test_retrieve_mixed_corpus_unstructured_query(mixed_retriever):
    results = mixed_retriever.retrieve("LLM pipeline RAG generation")
    texts = [d.page_content for d in results]
    assert any(
        "generation" in t.lower() or "pipeline" in t.lower()
        for t in texts
    )


def test_retrieve_mixed_corpus_structured_query(mixed_retriever):
    # Use values that appear verbatim after BM25 whitespace-split tokenisation
    # (keys like "Employee:" include the colon as part of the token)
    results = mixed_retriever.retrieve("Jane Doe Sales")
    texts = [d.page_content for d in results]
    assert any("Jane" in t or "Sales" in t for t in texts)


# ---------------------------------------------------------------------------
# 17–18  Single-document and edge-case corpora
# ---------------------------------------------------------------------------

def test_retrieve_single_document_corpus():
    single = [_doc("The only document in this corpus about retrieval")]
    with _PATCH:
        r = HybridRetriever(single)
    results = r.retrieve("retrieval document")
    assert len(results) == 1
    assert results[0].page_content == single[0].page_content


def test_retrieve_short_single_word_query(unstructured_retriever):
    results = unstructured_retriever.retrieve("FAISS")
    assert len(results) > 0


# ---------------------------------------------------------------------------
# 19–20  add_documents (dynamic ingestion)
# ---------------------------------------------------------------------------

def test_add_documents_increases_index_size(unstructured_retriever):
    original_count = len(unstructured_retriever.documents)
    new_docs = [
        _doc("New document about graph neural networks"),
        _doc("Another document about reinforcement learning"),
    ]
    with _PATCH:
        unstructured_retriever.add_documents(new_docs)
    assert len(unstructured_retriever.documents) == original_count + 2
    assert unstructured_retriever.bm25.corpus_size == original_count + 2


def test_add_documents_are_retrievable(unstructured_retriever):
    new_doc = _doc(
        "Zero-shot classification uses contrastive learning techniques"
    )
    with _PATCH:
        unstructured_retriever.add_documents([new_doc])
    results = unstructured_retriever.retrieve(
        "zero-shot contrastive classification"
    )
    texts = [d.page_content for d in results]
    assert any("contrastive" in t or "zero-shot" in t for t in texts)
