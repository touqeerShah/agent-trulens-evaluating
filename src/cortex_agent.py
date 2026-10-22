"""Legacy Snowflake Cortex Analyst/Search tool.

This module is intentionally separate from the local Ollama examples. The
AgentRunRequest model is submitted to Snowflake Cortex and cannot be replaced
with Ollama without changing the integration API.
"""

from __future__ import annotations

import json
from typing import Any, Type

from pydantic import BaseModel, PrivateAttr
from snowflake.core import Root
from snowflake.core.cortex.lite_agent_service import AgentRunRequest
from snowflake.snowpark import Session

SEMANTIC_MODEL_FILE = "@sales_intelligence.data.models/sales_metrics_model.yaml"
CORTEX_SEARCH_SERVICE = "sales_intelligence.data.sales_conversation_search"


class CortexAgentArgs(BaseModel):
    query: str


class CortexAgentTool:
    name: str = "CortexAgent"
    description: str = "answers questions using sales conversations and metrics"
    args_schema: Type[CortexAgentArgs] = CortexAgentArgs

    _session: Session = PrivateAttr()
    _root: Root = PrivateAttr()
    _agent_service: Any = PrivateAttr()

    def __init__(self, session: Session):
        self._session = session
        self._root = Root(session)
        self._agent_service = self._root.cortex_agent_service

    def _build_request(self, query: str) -> AgentRunRequest:
        return AgentRunRequest.from_dict(
            {
                # This is a Snowflake Cortex model, not the local Ollama model.
                "model": "claude-3-5-sonnet",
                "tools": [
                    {"tool_spec": {"type": "cortex_analyst_text_to_sql", "name": "analyst1"}},
                    {"tool_spec": {"type": "cortex_search", "name": "search1"}},
                ],
                "tool_resources": {
                    "analyst1": {"semantic_model_file": SEMANTIC_MODEL_FILE},
                    "search1": {
                        "name": CORTEX_SEARCH_SERVICE,
                        "max_results": 10,
                        "id_column": "conversation_id",
                    },
                },
                "messages": [{"role": "user", "content": [{"type": "text", "text": query}]}],
            }
        )

    def _consume_stream(self, stream):
        text, sql, citations = "", "", []
        for evt in stream.events():
            try:
                delta = (
                    evt.data.get("delta")
                    if isinstance(evt.data, dict)
                    else json.loads(evt.data).get("delta")
                    or json.loads(evt.data).get("data", {}).get("delta")
                )
            except Exception:
                continue
            if not isinstance(delta, dict):
                continue
            for item in delta.get("content", []):
                if item.get("type") == "text":
                    text += item.get("text", "")
                elif item.get("type") == "tool_results":
                    for result in item["tool_results"].get("content", []):
                        if result.get("type") != "json":
                            continue
                        payload = result["json"]
                        text += payload.get("text", "")
                        sql = payload.get("sql", sql)
                        citations.extend(
                            {"source_id": source.get("source_id"), "doc_id": source.get("doc_id")}
                            for source in payload.get("searchResults", [])
                        )
        return text, sql, str(citations)

    def run(self, query: str, **kwargs):
        request = self._build_request(query)
        stream = self._agent_service.run(request)
        text, sql, citations = self._consume_stream(stream)
        results_str = ""
        if sql:
            try:
                self._session.sql("USE WAREHOUSE SALES_INTELLIGENCE_WH").collect()
                results_str = self._session.sql(sql.rstrip(";")).to_pandas().to_string(index=False)
            except Exception as exc:
                results_str = f"SQL execution error: {exc}"
        return text, citations, sql, results_str
