# Agent evaluation practice with local Ollama

This repository now has two layers:

1. `src/helper.py` and `src/prompte.py` are the original, full Snowflake + LangGraph + TruLens lesson. They are intentionally kept as a legacy reference; none of their examples were deleted.
2. `src/ollama_practice/` is the local Ollama graph playground. It uses LangGraph and TruGraph for execution/tracing, Ollama for generation/evaluation, and Google Custom Search for web retrieval.

## Setup

Use Python 3.11.15 for the practice project. TruLens and its pinned dependencies are not currently compatible with the Python 3.14 environment that was previously selected.

Install/sync the project with `uv`:

```bash
uv sync
```

The project pins `setuptools<81` because this TruLens version imports `pkg_resources` through `munch`.
It also pins `trulens-feedback==2.2.1`; `trulens-core` records traces, while the
separate feedback package computes registered feedback functions in the
background evaluator.

Install and start Ollama, then pull a small model:

```bash
ollama serve
ollama pull gemma4:latest
```

Optional configuration:

```bash
export OLLAMA_MODEL=gemma4:latest
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_TEMPERATURE=0
export GOOGLE_API_KEY=your_google_api_key
export GOOGLE_CSE_ID=your_programmable_search_engine_id
```

The local TruLens retrieval feedbacks use OTEL selectors. The practice code
enables `TRULENS_OTEL_TRACING=1` automatically when it builds those feedbacks;
you do not need to configure an OpenAI or Anthropic evaluator.

You can also copy [.env.example](</Users/touqeershah/Documents/PharmaTraceProject-Files/deeplearning/agent-trulens-evaluating/.env.example>) to `.env`. The Snowflake variables are only needed by the legacy `src/helper.py` Snowflake/Cortex examples. The local graph examples need the Ollama variables plus `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` for web retrieval.

The graph uses Google’s Custom Search JSON API. Google requires both an API
key and a Programmable Search Engine ID (`cx`) for this endpoint. Review
Google’s current availability and transition notice before depending on this
API for production use.

Run the offline test without starting Ollama:

```bash
uv run python -m unittest discover -s tests -v
```

Edit the files in `src/ollama_practice/prompt_files/` to practice prompt changes without changing the graph code. Add or change questions in `src/ollama_practice/question_sets.py`.

## Separate QA cases

The original helper repeats these queries while changing the graph/evaluation setup. The same queries are available as small standalone files under `examples/qa_cases/`:

```bash
uv run python examples/qa_cases/01_market_cap.py
uv run python examples/qa_cases/02_regulatory_changes.py
uv run python examples/qa_cases/03_top_client_deals.py
uv run python examples/qa_cases/04_pending_deals_value_proposition.py
uv run python examples/qa_cases/05_largest_client_news.py
```

The Goal–Plan–Act examples are separate too:

```bash
uv run python examples/qa_cases/06_plan_quality.py
uv run python examples/qa_cases/07_plan_adherence.py
uv run python examples/qa_cases/08_execution_efficiency.py
uv run python examples/qa_cases/09_logical_consistency.py
```

Each file builds its graph directly, selects its predefined question, creates its TruGraph recorder, and runs the question. There is no `_runner.py` abstraction hiding the agent flow.

The pasted multi-version graph section is split under `examples/workflow_sets/`. Each file owns the full five-question set from the original code, its own graph configuration, and its own recorder/feedback configuration:

```bash
uv run python examples/workflow_sets/01_l4_base.py
uv run python examples/workflow_sets/02_l5_feedback.py
uv run python examples/workflow_sets/03_l6_subgoals.py
```

Shared code is limited to the Ollama client, question data, graph factory, prompt files, and local scoring. The L4/L5/L6 files do not import or execute `helper.py`.

## Suggested practice loop

1. Run the baseline and save the JSON output.
2. Change one prompt file only.
3. Run again and compare the scores and the actual answer.
4. Add a new question/reference pair before changing the evaluator.
5. Compare the TruGraph traces and local scores between L4, L5, and L6.

## Legacy example map

The original helper now contains only reusable agent/node and prompt/evaluation definitions. The runnable QA sets, graph construction, TruGraph recording, dashboards, and repeated queries were moved out into `examples/workflow_sets/` and `examples/qa_cases/`, so importing `src/helper.py` no longer runs those experiments.

## Where each practice lives

### Complete QA sets and graph versions

Each workflow file owns the complete five-question set, its graph configuration, and its feedback configuration. They do not import or execute `src/helper.py`.

| File | Practice | Graph difference | Feedback |
| --- | --- | --- | --- |
| `examples/workflow_sets/01_l4_base.py` | L4 Base | Basic planner, executor, research, chart, and synthesis nodes | Answer relevance, context relevance, groundedness |
| `examples/workflow_sets/02_l5_feedback.py` | L5 Goal-Plan-Act | Planner → researcher → executor → synthesizer | All seven Lesson 5 feedbacks: RAG triad plus GPA metrics |
| `examples/workflow_sets/03_l6_subgoals.py` | L6 Inline Evals + Sub-goals | Explicit goal/pre-condition/post-condition planning plus a bounded feedback-guided research loop | Same seven L5 feedbacks plus inline context feedback in graph state |

The shared five-question data is in `src/ollama_practice/question_sets.py`. The questions are:

1. Current market capitalization of the top US banks.
2. Current US financial-services regulatory changes.
3. Top three client deals and deal values.
4. Pending deals, regulatory changes, and value propositions.
5. Largest client deal, meeting topics, and related news.

Run a complete workflow version with:

```bash
uv run python examples/workflow_sets/01_l4_base.py
uv run python examples/workflow_sets/02_l5_feedback.py
uv run python examples/workflow_sets/03_l6_subgoals.py
```

The Lesson 5 file is the local equivalent of the original notebook's
`f_plan_quality`, `f_plan_adherence`, `f_execution_efficiency`, and
`f_logical_consistency` setup. It runs the three Lesson 5 recorded questions
through a real planner/researcher/executor/synthesizer graph and registers
those four GPA feedbacks together with the three RAG feedbacks.

Lesson 6 is implemented separately in
`src/ollama_practice/lesson6_graph.py`. Its planner prompt requires every
step to include an action, goal, pre-conditions, post-conditions, and expected
output. After retrieval, the graph runs an active local inline context check.
If evidence overlap with the goal is low, it performs at most one follow-up
research pass before executing the plan. The result is stored in
`inline_feedback` and is passed to the executor; it is not an unused
decorator. After execution, `validate_conditions` checks every planned
post-condition against the action trace and stores `condition_checks`,
`condition_score`, and `condition_failures` before synthesis.

### Individual QA examples

The same ideas are also available as one-question files under `examples/qa_cases/`:

| Files | Purpose |
| --- | --- |
| `01_market_cap.py` to `05_largest_client_news.py` | Run one research QA case at a time |
| `06_plan_quality.py` | Calls Ollama through LangGraph, records with TruGraph, and scores plan quality |
| `07_plan_adherence.py` | Calls Ollama through LangGraph, records with TruGraph, and compares plan/actions |
| `08_execution_efficiency.py` | Calls Ollama through LangGraph, records with TruGraph, and scores the trace |
| `09_logical_consistency.py` | Calls Ollama through LangGraph, records with TruGraph, and checks goal/action overlap |

For example:

```bash
uv run python examples/qa_cases/06_plan_quality.py
uv run python examples/qa_cases/09_logical_consistency.py
```

### Shared local implementation

- `src/ollama_practice/ollama_client.py` calls the local Ollama API.
- `src/ollama_practice/agent_graph.py` contains the real LangGraph nodes and graph factory used by the workflow-set files.
- `src/ollama_practice/google_search.py` calls Google Custom Search and formats returned title, snippet, and URL records as retrieval evidence.
- `src/ollama_practice/evaluate.py` contains transparent local relevance, groundedness, and evidence checks.
- `src/ollama_practice/prompt_files/` contains editable prompts.
- `src/ollama_practice/question_sets.py` contains the shared five-question dataset.

The local scores in `src/ollama_practice/evaluate.py` are provider-independent. They are intentionally small for practice and do not require an OpenAI or Anthropic API key.

The GPA examples use a real LangGraph in `src/ollama_practice/gpa_graph.py` with `planner`, `executor`, and `synthesizer` nodes. Each script calls the local `gemma4:latest` Ollama model, enters a real `TruGraph` recording context, and calculates its selected local score. The registered GPA feedbacks make a separate rubric-based Ollama judge call and return score/reason metadata; no GPT/Claude provider is configured.

The L4 workflow now mirrors the original Lesson 4 tracing sections:

- `agent_graph.py` instruments the researcher node as a retrieval span.
- The span records `QUERY_TEXT` and `RETRIEVED_CONTEXTS`.
- `trulens_feedbacks.py` attaches local Context Relevance, Groundedness, and Answer Relevance feedbacks to the graph recorder.
- `trulens_session.py` creates the OTEL-compatible SQLite TruLens session in `default_otel.sqlite`.
- `01_l4_base.py` runs the original three Lesson 4 QA cases from `LESSON4_QUESTIONS`.

Run the L4 trace and feedbacks:

```bash
uv run python examples/workflow_sets/01_l4_base.py
```

If you intentionally want to rebuild the OTEL practice database, use:

```bash
uv run python examples/workflow_sets/01_l4_base.py --reset-db
```

`--reset-db` is intentionally opt-in because it clears `default_otel.sqlite`.
The older `default.sqlite` file is left untouched. If an existing
`default_otel.sqlite` lacks the OTEL `trulens_events` table, the session helper
automatically preserves it and creates/uses `default_otel_v2.sqlite` instead.

Launch the optional dashboard after recording:

```bash
uv run python examples/workflow_sets/01_l4_base.py --dashboard
```

### Local TruLens feedback

The active RAG-triad feedback objects are built in `src/ollama_practice/trulens_feedbacks.py` and attached to every graph recorder. The named objects are:

- Groundedness
- Answer relevance
- Context relevance

The active RAG feedbacks use a separate Ollama evaluator call through
`src/ollama_practice/semantic_feedbacks.py`. The evaluator receives the
question, actual retrieval evidence, and answer, then returns a score, reason,
and unsupported claims. Missing required fields are marked
`not_evaluated` instead of being replaced with unrelated data.

The old word-overlap functions are retained as explicitly named diagnostics in
`feedbacks.py`; they are useful for baseline comparisons but are not presented
as semantic relevance or groundedness judgments. The feedback objects remain
registered with `TruGraph`, so their Ollama-generated reasons belong to the
recorded run.

The same module also builds the GPA TruLens feedback objects: `Plan Quality`, `Plan Adherence`, `Execution Efficiency`, and `Logical Consistency`. Each uses a separate rubric and evaluates the planner/executor output from the graph run through Ollama. The rubric score is normalized from 0–3 to 0–1 and includes reasons and improvement suggestions.

### Snowflake Cortex tool

The Snowflake-specific `AgentRunRequest` code is now isolated in `src/cortex_agent.py`. It contains:

- `CortexAgentTool`
- `_build_request()` with Cortex Analyst Text-to-SQL and Cortex Search tools
- Snowflake semantic-model and search-service configuration
- Cortex streaming/result handling

The request still contains `"model": "claude-3-5-sonnet"` because that model name is sent to Snowflake Cortex. It is not used by the local Ollama examples and is not changed to `gemma4:latest`; Ollama and Snowflake Cortex use different APIs. The legacy helper imports only `CortexAgentTool` from this module when the Snowflake workflow is used.

Note: the legacy module still requires its original cloud dependencies and environment variables. The new package does not import it, so local practice stays independent.
