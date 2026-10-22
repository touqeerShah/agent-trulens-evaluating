"""Run only the pending-deals question through the sub-goal graph."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from ollama_practice.agent_graph import build_graph, build_tru_recorder  # noqa: E402
from ollama_practice.evaluate import evaluate  # noqa: E402
from ollama_practice.feedbacks import build_feedbacks  # noqa: E402
from ollama_practice.ollama_client import OllamaClient  # noqa: E402
from ollama_practice.question_sets import QUESTIONS  # noqa: E402

case = QUESTIONS[3]
client = OllamaClient.from_env()
graph = build_graph("subgoals", client)
tru_recorder = build_tru_recorder(graph, "QA Pending Deals", build_feedbacks())
with tru_recorder:
    result = graph.invoke({"question": case["question"], "context": case["context"]})
print(json.dumps({"case": case["name"], "result": result, "scores": evaluate(result, case["question"], case["reference"])}, indent=2))
