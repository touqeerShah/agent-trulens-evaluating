"""Deterministic lexical diagnostics for the Ollama graph experiments.

These functions are intentionally simple diagnostics. They are not semantic
relevance or groundedness judges; the TruLens feedbacks use Ollama-backed
evaluators from :mod:`semantic_feedbacks`.
"""

from __future__ import annotations

import re


def _words(text: object) -> set[str]:
    return {word.lower() for word in re.findall(r"[a-zA-Z0-9]{4,}", str(text))}


def _value(payload: object, key: str, default: object = "") -> object:
    return payload.get(key, default) if isinstance(payload, dict) else default


def lexical_context_overlap(input_data: object = "", output_data: object = "") -> float:
    question_words = _words(_value(input_data, "question") or input_data)
    context = _value(input_data, "context")
    if not context and isinstance(output_data, (str, list, tuple)):
        context = output_data
    context_words = _words(context)
    return round(len(question_words & context_words) / max(len(question_words), 1), 3)


def lexical_groundedness_overlap(input_data: object = "", output_data: object = "") -> float:
    answer_words = _words(_value(output_data, "answer") or output_data)
    context = _value(input_data, "context")
    if not context and isinstance(input_data, (str, list, tuple)):
        context = input_data
    context_words = _words(context)
    return round(len(answer_words & context_words) / max(len(answer_words), 1), 3)


def lexical_answer_overlap(input_data: object = "", output_data: object = "") -> float:
    question_words = _words(_value(input_data, "question") or input_data)
    answer_words = _words(_value(output_data, "answer") or output_data)
    return round(len(question_words & answer_words) / max(len(question_words), 1), 3)


def build_feedbacks():
    """Build semantic TruLens feedbacks backed by the local Ollama model."""
    # Use the retrieval-span selectors so context feedback always receives
    # actual researcher output rather than model-written final-state evidence.
    from .trulens_feedbacks import build_rag_feedbacks

    return build_rag_feedbacks()
