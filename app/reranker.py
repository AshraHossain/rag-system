from sentence_transformers import CrossEncoder
from app.config import RERANKER_MODEL, TOP_K

# WHY:
# CrossEncoder scores query-document pairs jointly →
# far more accurate than bi-encoder or length heuristics.
# This is what production systems (Cohere, Jina) do under the hood.

_model = None


def get_reranker():
    """Lazy load — avoids reloading model on every request."""
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


def rerank(query, docs, top_k=None):
    if not docs:
        return []

    k = top_k or TOP_K
    model = get_reranker()

    pairs = [[query, doc.page_content] for doc in docs]
    scores = model.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:k]]