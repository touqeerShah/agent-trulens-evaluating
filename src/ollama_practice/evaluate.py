"""Provider-independent starter evaluations for comparing approaches."""

from __future__ import annotations

import re
from typing import Any


def _words(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[a-zA-Z0-9]{3,}", text)}


def evaluate(result: dict[str, Any], question: str, reference: str) -> dict[str, float]:
    """Return simple explainable scores in [0, 1]; replace these with TruLens later."""
    answer = str(result.get("answer", ""))
    question_words = _words(question)
    reference_words = _words(reference)
    answer_words = _words(answer)
    relevance = len(question_words & answer_words) / max(len(question_words), 1)
    groundedness = len(answer_words & reference_words) / max(len(answer_words), 1)
    has_evidence = bool(result.get("evidence"))
    return {"relevance": round(relevance, 3), "groundedness": round(groundedness, 3), "has_evidence": float(has_evidence)}


def compare(results: list[dict[str, Any]], question: str, reference: str) -> list[dict[str, Any]]:
    return [{**result, "scores": evaluate(result, question, reference)} for result in results]
