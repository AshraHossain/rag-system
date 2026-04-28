from types import SimpleNamespace

from app.evaluator import (
    answer_relevance,
    context_recall,
    evaluate_response,
    hallucination_score,
)


def _doc(text):
    return SimpleNamespace(page_content=text)


class TestAnswerRelevance:
    def test_exact_match(self):
        assert answer_relevance("what is rag", "rag is what") == 1.0

    def test_no_overlap(self):
        assert answer_relevance("apple banana", "car door") == 0.0

    def test_empty_query(self):
        assert answer_relevance("", "some answer") == 0.0

    def test_empty_answer(self):
        assert answer_relevance("some query", "") == 0.0

    def test_partial_overlap(self):
        score = answer_relevance("what is rag retrieval", "rag is cool")
        assert 0.0 < score < 1.0


class TestHallucinationScore:
    def test_fully_grounded(self):
        doc = _doc("retrieval augmented generation")
        score = hallucination_score("retrieval augmented generation", [doc])
        assert score == 0.0

    def test_fully_hallucinated(self):
        doc = _doc("apple banana cherry")
        score = hallucination_score("xyz uvw qrs", [doc])
        assert score == 1.0

    def test_empty_answer(self):
        assert hallucination_score("", [_doc("context")]) == 1.0


class TestContextRecall:
    def test_perfect_recall(self):
        doc = _doc("rag combines retrieval and generation")
        score = context_recall("rag combines retrieval and generation", [doc])
        assert score == 1.0

    def test_zero_recall(self):
        doc = _doc("apple banana cherry")
        score = context_recall("xyz uvw qrs", [doc])
        assert score == 0.0

    def test_empty_answer(self):
        assert context_recall("", [_doc("context")]) == 0.0


class TestEvaluateResponse:
    def test_returns_all_keys(self):
        docs = [_doc("retrieval augmented generation is useful")]
        result = evaluate_response(
            "what is rag",
            docs,
            "retrieval augmented generation",
        )
        assert set(result.keys()) == {
            "answer_relevance",
            "hallucination_score",
            "context_recall",
            "num_docs_used",
        }

    def test_num_docs_used(self):
        docs = [_doc("a"), _doc("b"), _doc("c")]
        result = evaluate_response("query", docs, "answer")
        assert result["num_docs_used"] == 3
