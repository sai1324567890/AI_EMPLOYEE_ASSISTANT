
import json
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .config import ROUTING_CONFIDENCE_THRESHOLD
from .retriever import Retriever

# Keyword priors: route -> list of trigger keywords/phrases (lowercase)
KEYWORD_HINTS = {
    "general_company": [
        "core value", "mission", "work hour", "work model", "remote work",
        "org structure", "organization structure", "company overview",
        "communication norm", "collaboration window", "common tools",
    ],
    "role_specific": [
        "data analyst", "product manager", "my role", "responsibilit",
        "first 30 days", "who should i ask", "who do i ask", "who to ask",
        "role expectation", "dashboard", "prd", "roadmap",
    ],
    "admin_policy": [
        "expense", "reimburse", "leave", "pto", "vacation", "timesheet",
        "attendance", "it ticket", "access request", "onboarding checklist",
        "security incident", "code of conduct", "compliance", "travel",
        "hr process", "verification letter", "policy",
    ],
}

# Phrases that should always be routed to direct_llm regardless of
# keyword/similarity overlap, because they involve personal/sensitive
# data or actions the assistant cannot perform from static documents.
DIRECT_LLM_OVERRIDES = [
    "my salary", "my pay", "my payroll", "tax deduction",
    "approve my leave", "approve my request", "hack", "bypass",
    "password of", "someone else's", "confidential employee data",
]


@dataclass
class RoutingDecision:
    route: str
    confidence: float
    scores: Dict[str, float]
    reason: str


def _keyword_score(query: str) -> Dict[str, float]:
    q = query.lower()
    scores = {r: 0.0 for r in KEYWORD_HINTS}
    for route, keywords in KEYWORD_HINTS.items():
        hits = sum(1 for kw in keywords if kw in q)
        if hits:
            scores[route] = min(1.0, 0.35 + 0.15 * hits)
    return scores


def classify(query: str, retriever: Retriever) -> RoutingDecision:
    q_lower = query.lower()

    for phrase in DIRECT_LLM_OVERRIDES:
        if phrase in q_lower:
            return RoutingDecision(
                route="direct_llm",
                confidence=1.0,
                scores={},
                reason=f"Matched sensitive/out-of-scope override phrase: '{phrase}'",
            )

    kw_scores = _keyword_score(query)
    tfidf_scores = retriever.score_routes(query)

    combined = {}
    for route in ("general_company", "role_specific", "admin_policy"):
        combined[route] = 0.5 * kw_scores.get(route, 0.0) + 0.5 * tfidf_scores.get(route, 0.0)

    best_route = max(combined, key=combined.get)
    best_score = combined[best_route]

    if best_score < ROUTING_CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            route="direct_llm",
            confidence=best_score,
            scores=combined,
            reason="No knowledge source cleared the confidence threshold; falling back to direct LLM.",
        )

    return RoutingDecision(
        route=best_route,
        confidence=best_score,
        scores=combined,
        reason=f"'{best_route}' scored highest (keyword+TF-IDF blend).",
    )


# ---------------------------------------------------------------------------
# LLM-based query classification (optional enhancement)
#
# Instead of the rule-based keyword+TF-IDF blend above, this asks the
# configured LLM backend to classify the question directly. It is used
# when config.ROUTER_BACKEND == "llm". If the LLM backend itself is
# "extractive" (no API key configured, i.e. there's no model to ask),
# this transparently falls back to the rule-based router so the app
# never breaks in a zero-config environment.
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM_PROMPT = """You are a query router for an employee onboarding assistant. \
Classify the employee's question into EXACTLY ONE of these routes:

- general_company: company mission/values, work hours/model, org structure, \
communication norms, tools used company-wide.
- role_specific: questions about a specific role's responsibilities, first 30/60/90 \
day expectations, who to ask for what, role-specific tools.
- admin_policy: expenses, leave/PTO, timesheets/attendance, IT access requests, \
onboarding checklist steps, travel, HR processes, code of conduct, security/compliance basics.
- direct_llm: anything involving personal data (salary, payroll), approvals the \
assistant cannot grant, security bypass attempts, or questions unrelated to onboarding.

Respond with ONLY a JSON object, no other text:
{"route": "<one of the four labels>", "confidence": <float 0-1>, "reason": "<short reason>"}
"""


def _parse_llm_route_json(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if data.get("route") not in ("general_company", "role_specific", "admin_policy", "direct_llm"):
        return None
    return data


def classify_llm(query: str, retriever: Retriever, llm_client) -> RoutingDecision:
    """LLM-based classification. Falls back to the rule-based `classify()`
    if the LLM backend is the offline `extractive` mode (nothing to ask)
    or if the LLM response can't be parsed."""
    q_lower = query.lower()
    for phrase in DIRECT_LLM_OVERRIDES:
        if phrase in q_lower:
            return RoutingDecision(
                route="direct_llm",
                confidence=1.0,
                scores={},
                reason=f"Matched sensitive/out-of-scope override phrase: '{phrase}'",
            )

    if getattr(llm_client, "backend", "extractive") == "extractive":
        decision = classify(query, retriever)
        decision.reason = "[llm router unavailable in offline/extractive mode -> rule_based] " + decision.reason
        return decision

    raw = llm_client.generate(
        system_prompt=_CLASSIFIER_SYSTEM_PROMPT,
        user_prompt=f"EMPLOYEE QUESTION:\n{query}",
        context_chunks=[],
    )
    parsed = _parse_llm_route_json(raw)
    if parsed is None:
        decision = classify(query, retriever)
        decision.reason = "[llm router response unparseable -> rule_based fallback] " + decision.reason
        return decision

    try:
        confidence = float(parsed.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.7

    return RoutingDecision(
        route=parsed["route"],
        confidence=confidence,
        scores={},
        reason=f"[llm router] {parsed.get('reason', '')}".strip(),
    )
