"""Local Ollama version of Lesson 6: sub-goal planning and inline research."""

from __future__ import annotations

import json
import re
from typing import Any

from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

from .feedbacks import lexical_context_overlap
from .google_search import GoogleSearchClient
from .ollama_client import OllamaClient
from .prompts import load_prompt


class Lesson6State(TypedDict, total=False):
    goal: str
    context: str
    plan: str
    evidence: list[str]
    actions: str
    answer: str
    research_round: int
    needs_more_research: bool
    inline_feedback: dict[str, Any]
    condition_checks: list[dict[str, Any]]
    condition_score: float
    condition_failures: list[str]


def build_lesson6_graph(
    client: OllamaClient | None = None,
    search_client: GoogleSearchClient | None = None,
):
    """Build the improved graph with one bounded feedback-guided retry."""
    client = client or OllamaClient.from_env()
    search_client = search_client or GoogleSearchClient.from_env()

    def planner(state: Lesson6State) -> dict[str, str]:
        def fallback_plan(action: str) -> str:
            return json.dumps(
                {
                    "steps": [
                        {
                            "action": action,
                            "goal": state["goal"],
                            "pre_conditions": [
                                "The user question and supplied context are available."
                            ],
                            "post_conditions": [
                                "A concrete result is produced and available to the next step."
                            ],
                            "expected_output": (
                                "A concrete result or an explicit missing-data note "
                                "that the next step can inspect."
                            ),
                        }
                    ]
                },
                indent=2,
            )

        raw = client.generate(
            load_prompt("planner_l6", question=state["goal"]), json_mode=True
        )
        try:
            plan = json.loads(raw)
            steps = plan["steps"]
            required = {
                "action",
                "goal",
                "pre_conditions",
                "post_conditions",
                "expected_output",
            }
            if not isinstance(steps, list) or not steps or any(
                not isinstance(step, dict) or not required.issubset(step)
                for step in steps
            ):
                raise ValueError("Planner returned an incomplete step schema")
            return {"plan": json.dumps(plan, indent=2)}
        except (json.JSONDecodeError, TypeError):
            return {"plan": fallback_plan(raw)}
        except (KeyError, ValueError):
            return {"plan": fallback_plan(raw)}

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes=lambda ret, exception, *args, **kwargs: {
            SpanAttributes.RETRIEVAL.QUERY_TEXT: args[0].get("goal", ""),
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: ret.get("evidence", []),
        },
    )
    def researcher(state: Lesson6State) -> dict[str, Any]:
        evidence = list(state.get("evidence", []))
        if not evidence:
            evidence.extend(search_client.search(state["goal"]))
        else:
            follow_up = client.generate(
                "Write one concise Google search query for a missing fact needed "
                "to answer this goal using the plan and current evidence. If "
                "nothing is missing, say NONE. Do not answer the question.\n\n"
                f"Goal: {state['goal']}\nPlan: {state.get('plan', '')}\n"
                f"Evidence: {' '.join(evidence)}"
            )
            if follow_up.strip().upper() != "NONE":
                evidence.extend(search_client.search(follow_up))
        return {
            "evidence": evidence,
            "research_round": state.get("research_round", 0) + 1,
        }

    def inline_research_eval(state: Lesson6State) -> dict[str, Any]:
        score = lexical_context_overlap(
            {"question": state["goal"]}, {"evidence": state.get("evidence", [])}
        )
        needs_more = score < 0.35 and state.get("research_round", 0) < 2
        return {
            "needs_more_research": needs_more,
            "inline_feedback": {
                "name": "Inline Context Relevance",
                "score": score,
                "reason": "Research again when evidence has insufficient goal overlap.",
                "action": "follow_up_research" if needs_more else "continue",
            },
        }

    def executor(state: Lesson6State) -> dict[str, str]:
        return {
            "actions": client.generate(
                "Execute the plan using the evidence and any inline feedback. "
                "Respect every pre-condition and post-condition. List numbered "
                "actions and identify missing information instead of inventing it.\n\n"
                f"Goal: {state['goal']}\nPlan: {state['plan']}\n"
                f"Evidence: {' '.join(state.get('evidence', []))}\n"
                f"Inline feedback: {state.get('inline_feedback', {})}"
            )
        }

    def validate_conditions(state: Lesson6State) -> dict[str, Any]:
        """Check each planned post-condition against the action trace."""
        try:
            steps = json.loads(state.get("plan", "{}"))["steps"]
        except (json.JSONDecodeError, KeyError, TypeError):
            steps = []

        action_text = state.get("actions", "").lower()
        checks: list[dict[str, Any]] = []
        failures: list[str] = []

        for index, step in enumerate(steps, 1):
            conditions = step.get("post_conditions", [])
            expected = step.get("expected_output", "")
            requirement_text = " ".join([*conditions, expected])
            terms = {
                token
                for token in re.findall(r"[a-zA-Z0-9]{4,}", requirement_text.lower())
                if token not in {"should", "must", "that", "this", "with", "from"}
            }
            matched = {term for term in terms if term in action_text}
            score = round(len(matched) / max(len(terms), 1), 3)
            passed = score >= 0.2
            check = {
                "step": index,
                "passed": passed,
                "score": score,
                "matched_terms": sorted(matched),
                "post_conditions": conditions,
            }
            checks.append(check)
            if not passed:
                failures.append(f"Step {index} post-conditions were not evidenced in actions.")

        return {
            "condition_checks": checks,
            "condition_score": round(
                sum(check["score"] for check in checks) / max(len(checks), 1), 3
            ),
            "condition_failures": failures,
        }

    def synthesizer(state: Lesson6State) -> dict[str, str]:
        return {
            "answer": client.generate(
                load_prompt(
                    "synthesizer",
                    question=state["goal"],
                    evidence="\n".join(
                        f"[{i}] {item}" for i, item in enumerate(state.get("evidence", []), 1)
                    )
                    + f"\n\nPlan: {state.get('plan', '')}"
                    + f"\n\nActions: {state.get('actions', '')}"
                    + f"\n\nCondition checks: {state.get('condition_checks', [])}"
                    + f"\n\nCondition failures: {state.get('condition_failures', [])}",
                )
            )
        }

    workflow = StateGraph(Lesson6State)
    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher)
    workflow.add_node("inline_research_eval", inline_research_eval)
    workflow.add_node("executor", executor)
    workflow.add_node("validate_conditions", validate_conditions)
    workflow.add_node("synthesizer", synthesizer)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "inline_research_eval")
    workflow.add_conditional_edges(
        "inline_research_eval",
        lambda state: "researcher" if state.get("needs_more_research") else "executor",
        {"researcher": "researcher", "executor": "executor"},
    )
    workflow.add_edge("executor", "validate_conditions")
    workflow.add_edge("validate_conditions", "synthesizer")
    workflow.add_edge("synthesizer", END)
    return workflow.compile()
