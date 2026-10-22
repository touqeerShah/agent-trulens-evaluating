"""Local RAG-triad feedbacks with the same selector shape as Lesson 4."""

from __future__ import annotations

import os

# The retrieval selectors below are TruLens OTEL selectors.  Enable the mode
# before constructing Feedback objects; otherwise TruLens expects legacy Lens
# selectors and rejects Selector instances.
os.environ.setdefault("TRULENS_OTEL_TRACING", "1")

from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes

from .semantic_feedbacks import (
    semantic_answer_relevance,
    semantic_context_relevance,
    semantic_groundedness,
    build_semantic_gpa_feedbacks,
)


def build_rag_feedbacks():
    """Attach local RAG-triad functions to retrieval spans and final output."""
    f_groundedness = (
        Feedback(semantic_groundedness, name="Groundedness")
        .on(
            {
                "input_data": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                    collect_list=True,
                )
            }
        )
        .on_output()
    )
    f_answer_relevance = Feedback(semantic_answer_relevance, name="Answer Relevance").on_input().on_output()
    f_context_relevance = (
        Feedback(semantic_context_relevance, name="Context Relevance")
        .on(
            {
                "input_data": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.QUERY_TEXT,
                )
            }
        )
        .on(
            {
                "output_data": Selector(
                    span_type=SpanAttributes.SpanType.RETRIEVAL,
                    span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
                    collect_list=True,
                )
            }
        )
    )
    return [f_answer_relevance, f_context_relevance, f_groundedness]


def build_gpa_feedbacks():
    """Build Ollama-judged GPA feedbacks with rubric-based reasons."""
    return build_semantic_gpa_feedbacks()
