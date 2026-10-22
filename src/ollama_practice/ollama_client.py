"""A tiny Ollama client so the examples work without OpenAI or Anthropic."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class OllamaClient:
    """Call a locally running Ollama server through its HTTP API."""

    model: str = "gemma4:latest"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0

    @classmethod
    def from_env(cls) -> "OllamaClient":
        return cls(
            model=os.getenv("OLLAMA_MODEL", "gemma4:latest"),
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0")),
        )

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Ollama is not reachable. Start it with `ollama serve` and pull "
                f"the model with `ollama pull {self.model}`."
            ) from exc
        return str(result.get("response", "")).strip()

