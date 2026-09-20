"""
Thin client for Serper.dev (a Google Search API wrapper).

Used as a fallback source when a question doesn't match anything in the
local onboarding knowledge base. This module never raises on network/API
failures — it always returns a `WebSearchResponse`, with `.ok=False` and a
human-readable `.error` if something went wrong, so it can never take down
the chat flow the way a bad LLM call could.
"""
import json
from dataclasses import dataclass, field
from typing import List
from urllib import request as urlrequest
from urllib.error import HTTPError, URLError

from . import config


@dataclass
class WebResult:
    title: str
    link: str
    snippet: str


@dataclass
class WebSearchResponse:
    ok: bool
    query: str
    results: List[WebResult] = field(default_factory=list)
    answer_box: str = ""       # Serper's direct "answer box" snippet, if present
    error: str = ""


class WebSearchClient:
    def __init__(self):
        self.enabled = config.WEB_SEARCH_ENABLED
        self.api_key = config.SERPER_API_KEY

    def search(self, query: str, num: int = None) -> WebSearchResponse:
        if not self.enabled:
            return WebSearchResponse(ok=False, query=query, error="Web search is not configured.")

        num = num or config.WEB_SEARCH_RESULTS
        payload = json.dumps({"q": query, "num": num}).encode("utf-8")
        req = urlrequest.Request(
            config.SERPER_API_URL,
            data=payload,
            headers={
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlrequest.urlopen(req, timeout=config.WEB_SEARCH_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8")[:300]
            except Exception:
                pass
            if e.code == 401 or e.code == 403:
                reason = "Serper API key was rejected (invalid, revoked, or out of credits)."
            else:
                reason = f"Serper API returned HTTP {e.code}."
            return WebSearchResponse(ok=False, query=query, error=f"{reason} {detail}".strip())
        except URLError as e:
            return WebSearchResponse(ok=False, query=query, error=f"Could not reach Serper API: {e.reason}")
        except Exception as e:
            return WebSearchResponse(ok=False, query=query, error=f"Unexpected web search error: {type(e).__name__}: {e}")

        results = []
        for item in body.get("organic", [])[:num]:
            results.append(WebResult(
                title=item.get("title", "").strip(),
                link=item.get("link", "").strip(),
                snippet=item.get("snippet", "").strip(),
            ))

        answer_box = ""
        ab = body.get("answerBox")
        if isinstance(ab, dict):
            answer_box = (ab.get("answer") or ab.get("snippet") or "").strip()

        if not results and not answer_box:
            return WebSearchResponse(ok=False, query=query, error="No web results found for this query.")

        return WebSearchResponse(ok=True, query=query, results=results, answer_box=answer_box)
