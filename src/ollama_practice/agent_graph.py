"""The actual experiment graph used by the local Ollama runs.

Each experiment chooses a flow:

* ``direct``: researcher -> synthesizer
* ``structured``: researcher -> structured-answer synthesizer
* ``subgoals``: planner -> researcher -> synthesizer

The experiment files own the question loop and TruGraph recorder. This module
only contains reusable graph nodes and graph construction.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from .ollama_client import OllamaClient
from .prompts import load_prompt
from .google_search import GoogleSearchClient


class AgentState(TypedDict, total=False):
    question: str
    context: str
    plan: dict[str, Any]
    answer: str
    evidence: list[str]


def _planner(client: OllamaClient, state: AgentState) -> dict[str, Any]:
    raw = client.generate(load_prompt("planner", question=state["question"]), json_mode=True)
    try:
        return {"plan": json.loads(raw)}
    except json.JSONDecodeError:
        return {"plan": {"steps": [raw]}}


@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("question", ""),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: ret.get("evidence", []),
    },
)
def _researcher(state: AgentState, search_client: GoogleSearchClient) -> dict[str, Any]:
    """Retrieve real web evidence from Google for the current question."""
    return {"evidence": search_client.search(state["question"])}


def _direct_synthesizer(client: OllamaClient, state: AgentState) -> dict[str, str]:
    answer = client.generate(
        load_prompt("direct", question=state["question"], context="\n".join(state["evidence"]))
    )
    return {"answer": answer}


def _structured_synthesizer(client: OllamaClient, state: AgentState) -> dict[str, Any]:
    raw = client.generate(
        load_prompt("structured", question=state["question"], context="\n".join(state["evidence"])),
        json_mode=True,
    )
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"answer": raw, "evidence": state["evidence"], "confidence": 0}
    return {"answer": str(result.get("answer", raw)), "evidence": result.get("evidence", state["evidence"])}


def _planned_synthesizer(client: OllamaClient, state: AgentState) -> dict[str, str]:
    answer = client.generate(
        load_prompt(
            "synthesizer",
            question=state["question"],
            evidence="\n".join(f"[{i}] {item}" for i, item in enumerate(state["evidence"], 1))
            + f"\n\nPlan used: {state.get('plan', {})}",
        )
    )
    return {"answer": answer}


def build_graph(
    flow: Literal["direct", "structured", "subgoals"],
    client: OllamaClient | None = None,
    search_client: GoogleSearchClient | None = None,
):
    """Build one independent agent graph for an experiment."""
    client = client or OllamaClient.from_env()
    search_client = search_client or GoogleSearchClient.from_env()
    workflow = StateGraph(AgentState)

    if flow == "subgoals":
        workflow.add_node("planner", lambda state: _planner(client, state))
        workflow.add_edge(START, "planner")
        workflow.add_node("researcher", lambda state: _researcher(state, search_client))
        workflow.add_edge("planner", "researcher")
        workflow.add_node("synthesizer", lambda state: _planned_synthesizer(client, state))
        workflow.add_edge("researcher", "synthesizer")
    else:
        workflow.add_node("researcher", lambda state: _researcher(state, search_client))
        workflow.add_edge(START, "researcher")
        synthesizer = _structured_synthesizer if flow == "structured" else _direct_synthesizer
        workflow.add_node("synthesizer", lambda state: synthesizer(client, state))
        workflow.add_edge("researcher", "synthesizer")

    workflow.add_edge("synthesizer", END)
    return workflow.compile()


def build_tru_recorder(graph, name: str, feedbacks=None):
    """Build the real TruGraph recorder used by every experiment."""
    from trulens.apps.langgraph import TruGraph

    return TruGraph(graph, app_name=name, app_version="ollama-local", feedbacks=feedbacks or [])
