"""A local Ollama + LangGraph workflow for Goal-Plan-Act practice."""

from __future__ import annotations

import json
import re
from typing import Any

from typing_extensions import TypedDict

from langgraph.graph import END, START, StateGraph

from .ollama_client import OllamaClient


class GPAState(TypedDict, total=False):
    goal: str
    plan: str
    actions: str
    answer: str


def build_gpa_graph(client: OllamaClient | None = None):
    client = client or OllamaClient.from_env()

    def planner(state: GPAState) -> dict[str, str]:
        raw = client.generate(
            "Create a concise execution plan. Return JSON only with a `plan` "
            "string containing numbered steps.\n\nGoal: " + state["goal"],
            json_mode=True,
        )
        try:
            plan = str(json.loads(raw)["plan"])
        except (json.JSONDecodeError, KeyError, TypeError):
            plan = raw
        return {"plan": plan}

    def executor(state: GPAState) -> dict[str, str]:
        return {
            "actions": client.generate(
                "Execute this plan conceptually. Describe actions step by step and "
                "do not invent tool results.\n\n"
                f"Goal: {state['goal']}\nPlan: {state['plan']}"
            )
        }

    def synthesizer(state: GPAState) -> dict[str, str]:
        return {
            "answer": client.generate(
                "Summarize whether the actions satisfy the goal and mention missing "
                "work.\n\n"
                f"Goal: {state['goal']}\nPlan: {state['plan']}\nActions: {state['actions']}"
            )
        }

    workflow = StateGraph(GPAState)
    workflow.add_node("planner", planner)
    workflow.add_node("executor", executor)
    workflow.add_node("synthesizer", synthesizer)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()


def build_tru_graph_recorder(graph, name: str, feedbacks=None):
    """Create TruGraph for tracing without a remote feedback provider."""
    try:
        from trulens.apps.langgraph import TruGraph
    except ImportError as exc:
        raise RuntimeError("Run `uv sync` before running GPA examples.") from exc
    return TruGraph(graph, app_name=name, app_version="ollama-local", feedbacks=feedbacks or [])


def _words(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[a-zA-Z0-9]{4,}", text)}


def score_gpa(state: GPAState, metric: str) -> float:
    plan_words = _words(state.get("plan", ""))
    action_words = _words(state.get("actions", ""))
    if metric == "plan_quality":
        return round(min(1.0, len(plan_words) / 20), 3)
    if metric in {"plan_adherence", "logical_consistency"}:
        return round(len(plan_words & action_words) / max(len(plan_words), 1), 3)
    if metric == "execution_efficiency":
        return round(min(1.0, len(action_words) / max(len(plan_words) * 2, 1)), 3)
    raise ValueError(f"Unknown GPA metric: {metric}")


def run_gpa_case(goal: str, metric: str, recorder_name: str) -> dict[str, Any]:
    client = OllamaClient.from_env()
    graph = build_gpa_graph(client)
    from .trulens_feedbacks import build_gpa_feedbacks

    recorder = build_tru_graph_recorder(graph, recorder_name, build_gpa_feedbacks())
    with recorder:
        final_state = graph.invoke({"goal": goal})
    return {
        "metric": metric,
        "model": client.model,
        "graph_nodes": ["planner", "executor", "synthesizer"],
        "tru_recorder": type(recorder).__name__,
        "plan": final_state.get("plan", ""),
        "actions": final_state.get("actions", ""),
        "answer": final_state.get("answer", ""),
        "score": score_gpa(final_state, metric),
    }
