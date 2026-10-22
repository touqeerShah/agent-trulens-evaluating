"""Local Ollama version of the Lesson 5 Goal-Plan-Act graph."""

from __future__ import annotations

import json
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from .ollama_client import OllamaClient
from .prompts import load_prompt
from .google_search import GoogleSearchClient


class Lesson5State(TypedDict, total=False):
    goal: str
    context: str
    plan: str
    evidence: list[str]
    actions: str
    answer: str


def build_lesson5_graph(
    client: OllamaClient | None = None,
    search_client: GoogleSearchClient | None = None,
):
    """Build the Lesson 5 planner/researcher/executor/synthesizer graph."""
    client = client or OllamaClient.from_env()
    search_client = search_client or GoogleSearchClient.from_env()

    def planner(state: Lesson5State) -> dict[str, str]:
        raw = client.generate(
            "Create a specific, ordered plan for this goal. Include filters, "
            "prioritization criteria, required evidence, and the requested output. "
            "Return JSON only with a `plan` string.\n\nGoal: " + state["goal"],
            json_mode=True,
        )
        try:
            return {"plan": str(json.loads(raw)["plan"])}
        except (json.JSONDecodeError, KeyError, TypeError):
            return {"plan": raw}

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("goal", ""),
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: ret.get("evidence", []),
        },
    )
    def researcher(state: Lesson5State) -> dict[str, Any]:
        return {"evidence": search_client.search(state["goal"])}

    def executor(state: Lesson5State) -> dict[str, str]:
        return {
            "actions": client.generate(
                "Follow the plan exactly using only the supplied evidence. "
                "Write numbered actions and explicitly state any missing data. "
                "Do not invent tool results.\n\n"
                f"Goal: {state['goal']}\nPlan: {state['plan']}\n"
                f"Evidence: {' '.join(state.get('evidence', []))}"
            )
        }

    def synthesizer(state: Lesson5State) -> dict[str, str]:
        return {
            "answer": client.generate(
                load_prompt(
                    "synthesizer",
                    question=state["goal"],
                    evidence="\n".join(state.get("evidence", []))
                    + f"\n\nPlan: {state.get('plan', '')}"
                    + f"\n\nActions: {state.get('actions', '')}",
                )
            )
        }

    workflow = StateGraph(Lesson5State)
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("executor", executor)
    workflow.add_node("synthesizer", synthesizer)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()

