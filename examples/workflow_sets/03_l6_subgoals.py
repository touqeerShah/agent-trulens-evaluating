"""Lesson 6: improve GPA with explicit sub-goals and inline research feedback."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ollama_practice.agent_graph import build_tru_recorder  # noqa: E402
from ollama_practice.evaluate import evaluate  # noqa: E402
from ollama_practice.lesson6_graph import build_lesson6_graph  # noqa: E402
from ollama_practice.ollama_client import OllamaClient  # noqa: E402
from ollama_practice.question_sets import LESSON4_QUESTIONS  # noqa: E402
from ollama_practice.trulens_feedbacks import build_gpa_feedbacks, build_rag_feedbacks  # noqa: E402
from ollama_practice.trulens_session import create_session, launch_dashboard  # noqa: E402


client = OllamaClient.from_env()
graph = build_lesson6_graph(client)
create_session(reset="--reset-db" in sys.argv)
tru_recorder = build_tru_recorder(
    graph,
    "L6 Inline Evals and Sub-goals Ollama Agent",
    build_rag_feedbacks() + build_gpa_feedbacks(),
)

if __name__ == "__main__":
    with tru_recorder:
        for case in LESSON4_QUESTIONS:
            result = graph.invoke({"goal": case["question"]})
            print(json.dumps({"flow": "L6 Inline Evals + Sub-goals", "case": case["name"], "result": result, "scores": evaluate(result, case["question"], case["reference"])}, indent=2))
    if "--dashboard" in sys.argv:
        launch_dashboard()
