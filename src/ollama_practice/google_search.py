"""Google Programmable Search retrieval for the local graph experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GoogleSearchClient:
    """Call Google's Custom Search JSON API and return traceable evidence."""

    api_key: str
    search_engine_id: str
    max_results: int = 5

    @classmethod
    def from_env(cls) -> "GoogleSearchClient":
        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        search_engine_id = os.getenv("GOOGLE_CSE_ID", "").strip()
        if not api_key or not search_engine_id:
            raise RuntimeError(
                "Google Search is not configured. Set GOOGLE_API_KEY and "
                "GOOGLE_CSE_ID in .env or the shell environment."
            )
        return cls(
            api_key=api_key,
            search_engine_id=search_engine_id,
            max_results=int(os.getenv("GOOGLE_SEARCH_RESULTS", "5")),
        )

    def search(self, query: str) -> list[str]:
        params = urlencode(
            {
                "key": self.api_key,
                "cx": self.search_engine_id,
                "q": query,
                "num": min(max(self.max_results, 1), 10),
            }
        )
        request = Request(
            f"https://www.googleapis.com/customsearch/v1?{params}",
            headers={"Accept": "application/json"},
        )
        try:
            with urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Google Search API request failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError("Google Search API is not reachable.") from exc

        evidence = []
        for item in payload.get("items", []):
            title = item.get("title", "Untitled result")
            snippet = item.get("snippet", "")
            link = item.get("link", "")
            evidence.append(f"{title}\n{snippet}\nSource: {link}")
        return evidence
