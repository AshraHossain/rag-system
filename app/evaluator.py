# WHY THIS FILE EXISTS:
# Most candidates DON'T measure quality → you will.


def precision_at_k(retrieved_docs, relevant_docs):
    retrieved_texts = [doc.page_content for doc in retrieved_docs]
    relevant_texts = relevant_docs

    hits = len(set(retrieved_texts) & set(relevant_texts))
    return hits / len(retrieved_texts) if retrieved_texts else 0


def hallucination_score(answer, context_docs):
    """
    VERY simple heuristic:
    Checks how much of the answer overlaps with context
    """

    context = " ".join([doc.page_content for doc in context_docs])

    overlap = sum(1 for word in answer.split() if word in context.split())

    return 1 - (overlap / len(answer.split()) if answer.split() else 0)


def answer_relevance(query, answer):
    """
    Measures if the answer is topically related to the query.
    Uses simple word overlap — replace with embeddings in production.
    """
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())
    
    if not answer_words or not query_words:
        return 0.0
    
    overlap = len(query_words & answer_words)
    return round(overlap / len(query_words), 3)


def evaluate_response(query, docs, answer):
    dummy_relevant = [doc.page_content for doc in docs[:2]]
    precision = precision_at_k(docs, dummy_relevant)
    hallucination = hallucination_score(answer, docs)
    relevance = answer_relevance(query, answer)

    return {
        "precision_at_k": round(precision, 3),
        "hallucination_score": round(hallucination, 3),
        "answer_relevance": relevance,
        "num_docs_used": len(docs)
    }