"""L4 experiment: run the basic agent flow over every predefined question."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ollama_practice.agent_graph import build_graph, build_tru_recorder  # noqa: E402
from ollama_practice.evaluate import evaluate  # noqa: E402
from ollama_practice.ollama_client import OllamaClient  # noqa: E402
from ollama_practice.question_sets import LESSON4_QUESTIONS  # noqa: E402
from ollama_practice.trulens_feedbacks import build_rag_feedbacks  # noqa: E402
from ollama_practice.trulens_session import create_session, launch_dashboard  # noqa: E402


client = OllamaClient.from_env()
graph = build_graph("direct", client)
session = create_session(reset="--reset-db" in sys.argv)
tru_recorder = build_tru_recorder(graph, "L4 Base Ollama Agent", build_rag_feedbacks())

if __name__ == "__main__":
    with tru_recorder:
        for case in LESSON4_QUESTIONS:
            result = graph.invoke({"question": case["question"]})
            print(json.dumps({"flow": "L4 Base", "case": case["name"], "result": result, "scores": evaluate(result, case["question"], case["reference"])}, indent=2))
    if "--dashboard" in sys.argv:
        launch_dashboard()
