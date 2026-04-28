from typing import Any, cast


def hallucination_score(answer: str, context_docs: list) -> float:
    context_words = set()
    for doc in context_docs:
        context_words.update(doc.page_content.lower().split())
    answer_words = answer.lower().split()
    if not answer_words:
        return 1.0
    overlap = sum(1 for w in answer_words if w in context_words)
    return round(1.0 - overlap / len(answer_words), 3)


def answer_relevance(query: str, answer: str) -> float:
    q_words = set(query.lower().split())
    a_words = set(answer.lower().split())
    if not q_words or not a_words:
        return 0.0
    return round(len(q_words & a_words) / len(q_words), 3)


def context_recall(answer: str, docs: list) -> float:
    a_words = set(answer.lower().split())
    if not a_words:
        return 0.0
    ctx_words = set()
    for doc in docs:
        ctx_words.update(doc.page_content.lower().split())
    return round(len(a_words & ctx_words) / len(a_words), 3)


def evaluate_response(query: str, docs: list, answer: str) -> dict:
    return {
        "answer_relevance": answer_relevance(query, answer),
        "hallucination_score": hallucination_score(answer, docs),
        "context_recall": context_recall(answer, docs),
        "num_docs_used": len(docs),
    }


def run_ragas_evaluation(
    questions: list,
    answers: list,
    contexts: list,
    ground_truths: list | None = None,
) -> dict:
    """Batch RAGas evaluation. Requires OPENAI_API_KEY in environment."""
    try:
        from datasets import Dataset
        from ragas import evaluate

        try:
            from ragas.metrics import Faithfulness, AnswerRelevancy
            metrics = [Faithfulness(), AnswerRelevancy()]
            if ground_truths:
                from ragas.metrics import ContextRecall
                metrics.append(ContextRecall())
        except ImportError:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
            )
            metrics = [faithfulness, answer_relevancy]
            if ground_truths:
                from ragas.metrics import context_recall
                metrics.append(context_recall)

        data: dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
        }
        if ground_truths:
            data["ground_truth"] = ground_truths

        result = cast(Any, evaluate(Dataset.from_dict(data), metrics=metrics))
        if hasattr(result, "to_pandas"):
            raw = result.to_pandas().mean(numeric_only=True).to_dict()
        elif hasattr(result, "scores"):
            merged: dict = {}
            for row in result.scores:
                for k, v in row.items():
                    merged.setdefault(k, []).append(v)
            raw = {k: sum(v) / len(v) for k, v in merged.items()}
        else:
            raw = vars(result)
        return {
            k: round(float(v), 4)
            for k, v in raw.items()
            if isinstance(v, (int, float))
        }
    except Exception as exc:
        return {
            "error": str(exc),
            "message": (
                "RAGas evaluation failed. "
                "Ensure OPENAI_API_KEY is set and ragas is installed."
            ),
        }
