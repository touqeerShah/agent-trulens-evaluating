"""GPA 4: check whether executor actions support the stated goal."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from ollama_practice.gpa_graph import run_gpa_case  # noqa: E402

print(json.dumps(run_gpa_case("Which sales leads should we prioritize this week?", "logical_consistency", "GPA Logical Consistency"), indent=2))
