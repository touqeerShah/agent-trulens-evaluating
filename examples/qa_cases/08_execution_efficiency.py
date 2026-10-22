"""GPA 3: inspect local graph execution for unnecessary work."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ollama_practice.gpa_graph import run_gpa_case  # noqa: E402

print(json.dumps(run_gpa_case("Which sales leads should we prioritize this week?", "execution_efficiency", "GPA Execution Efficiency"), indent=2))
