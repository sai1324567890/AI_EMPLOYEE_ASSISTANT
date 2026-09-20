
from dataclasses import dataclass, field
from typing import List, Optional

from . import config
from .document_loader import load_corpus, Chunk
from .retriever import Retriever, RetrievedChunk
from .router import classify, classify_llm, RoutingDecision
from .llm_client import LLMClient
from .memory import SessionMemoryStore
from .web_search import WebSearchClient
from .prompts import (
    SYSTEM_PROMPT_RAG,
    SYSTEM_PROMPT_DIRECT,
    SYSTEM_PROMPT_WEB,
    build_rag_user_prompt,
    build_direct_user_prompt,
    build_web_user_prompt,
)

try:
    from .vector_retriever import VectorRetriever
    _FAISS_AVAILABLE = True
except ImportError:
    _FAISS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Follow-up detection
#
# Retrieval matches chunks against the literal text of the CURRENT question.
# That breaks the moment someone asks a short follow-up like "explain in
# detail" or "simpler please" — those words alone don't retrieve anything
# about the original topic. So: if a question looks like a follow-up, we
# fold the previous question's topic into the search query (not into what
# gets shown to the LLM as "the question" — that stays exactly what the
# person typed) and pull a few extra chunks so there's enough material to
# actually elaborate on.
# ---------------------------------------------------------------------------
FOLLOWUP_TRIGGER_PHRASES = [
    "explain in detail", "in detail", "more detail", "more details", "elaborate",
    "explain more", "tell me more", "go deeper", "expand on that", "expand on this",
    "simplify", "simpler", "in simple terms", "in simple words", "simple way",
    "explain again", "what do you mean", "clarify", "break it down", "break that down",
    "summarize that", "summarise that", "explain this better", "more info",
    "can you explain that", "why", "how so", "give an example", "for example",
]


def _looks_like_followup(question: str, memory) -> bool:
    """Heuristic: explicit follow-up phrasing, or a short question typed
    right after a previous turn (which is almost always referring back to
    it rather than starting a new topic from scratch)."""
    if memory.is_empty():
        return False
    q = question.lower().strip().rstrip("?.!")
    if any(p in q for p in FOLLOWUP_TRIGGER_PHRASES):
        return True
    if len(q.split()) <= 5:
        return True
    return False


def _build_retrieval_query(question: str, memory, is_followup: bool) -> str:
    """The query used for ROUTING + RETRIEVAL only. The LLM always still
    sees the person's actual question verbatim, plus the conversation
    history block — this just makes sure the *right documents* get pulled
    back in for a short follow-up."""
    if not is_followup:
        return question
    last_question = memory.turns[-1].question
    return f"{last_question} {question}"


@dataclass
class AssistantResult:
    question: str
    route: str
    routing_confidence: float
    routing_scores: dict
    routing_reason: str = ""
    retrieved: List[RetrievedChunk] = field(default_factory=list)
    web_sources: List[dict] = field(default_factory=list)  # [{"title", "link"}]
    answer: str = ""
    backend: str = ""
    retrieval_backend: str = ""
    router_backend: str = ""
    query_id: str = ""
    used_followup_memory: bool = False

    @property
    def citations(self) -> List[str]:
        if self.web_sources:
            return [w["link"] for w in self.web_sources]
        return [rc.chunk.citation for rc in self.retrieved]


class TrainingAssistant:
    def __init__(self, retrieval_backend: Optional[str] = None,
                 router_backend: Optional[str] = None,
                 log_analytics: bool = True):
        self.chunks: List[Chunk] = load_corpus(config.ROUTE_SOURCES)

        self.retrieval_backend = retrieval_backend or config.RETRIEVAL_BACKEND
        if self.retrieval_backend == "faiss" and _FAISS_AVAILABLE:
            self.retriever = VectorRetriever(self.chunks, embedding_dim=config.FAISS_EMBEDDING_DIM)
        else:
            if self.retrieval_backend == "faiss" and not _FAISS_AVAILABLE:
                self.retrieval_backend = "tfidf"  # faiss package not installed -> fall back
            self.retriever = Retriever(self.chunks)

        self.router_backend = router_backend or config.ROUTER_BACKEND
        self.llm = LLMClient()
        self.web_search = WebSearchClient()
        self.memory = SessionMemoryStore(max_turns=config.CONVERSATION_MEMORY_TURNS)
        self.log_analytics = log_analytics

    def route_only(self, question: str) -> RoutingDecision:
        if self.router_backend == "llm":
            return classify_llm(question, self.retriever, self.llm)
        return classify(question, self.retriever)

    def reset_conversation(self, session_id: str = "default") -> None:
        self.memory.reset(session_id)

    def ask(self, question: str, top_k: int = None, session_id: str = "default",
            use_memory: bool = True) -> AssistantResult:
        top_k = top_k or config.TOP_K
        memory = self.memory.get(session_id)
        history_block = memory.as_prompt_block() if (use_memory and not memory.is_empty()) else ""

        is_followup = use_memory and _looks_like_followup(question, memory)
        retrieval_query = _build_retrieval_query(question, memory, is_followup) if use_memory else question
        # A follow-up asking to elaborate needs more material to work with
        # than the original question did.
        retrieval_top_k = min(top_k + 2, top_k * 2) if is_followup else top_k

        decision = self.route_only(retrieval_query)

        # The rule-based router already gates document routes behind
        # ROUTING_CONFIDENCE_THRESHOLD before ever calling the retriever, so
        # an unrelated question correctly falls through to direct_llm (and
        # then web search). classify_llm() has no equivalent check — it
        # trusts whatever route the LLM picks, and since TF-IDF/FAISS
        # similarity is almost never *exactly* zero, retriever.top_k() would
        # then return "matching" chunks for basically any query, silently
        # answering from irrelevant documents instead of ever trying web
        # search. Re-checking the actual retrieval similarity here — for
        # BOTH router backends — closes that gap and keeps behavior
        # consistent regardless of which router picked the route.
        if decision.route != "direct_llm":
            route_scores = self.retriever.score_routes(retrieval_query)
            actual_score = route_scores.get(decision.route, 0.0)
            if actual_score < config.ROUTING_CONFIDENCE_THRESHOLD:
                decision = RoutingDecision(
                    route="direct_llm",
                    confidence=actual_score,
                    scores=route_scores,
                    reason=(
                        decision.reason
                        + f" [overridden: retrieval similarity for '{decision.route}' "
                        f"was {actual_score:.2f}, below the {config.ROUTING_CONFIDENCE_THRESHOLD} "
                        "confidence threshold -> falling back to direct_llm/web search]"
                    ),
                )

        if decision.route == "direct_llm":
            result = self._answer_out_of_scope(question, history_block, decision, decision.reason)
            result.used_followup_memory = is_followup
            self._finalize(result, memory, session_id)
            return result

        retrieved = self.retriever.top_k(retrieval_query, decision.route, k=retrieval_top_k)

        if not retrieved:
            reason = decision.reason + " (no chunks retrieved -> checking web/direct_llm fallback)"
            result = self._answer_out_of_scope(question, history_block, decision, reason)
            result.used_followup_memory = is_followup
            self._finalize(result, memory, session_id)
            return result

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT_RAG,
            user_prompt=build_rag_user_prompt(question, retrieved, history_block, is_followup=is_followup),
            context_chunks=retrieved,
        )
        result = AssistantResult(
            question=question,
            route=decision.route,
            routing_confidence=decision.confidence,
            routing_scores=decision.scores,
            routing_reason=decision.reason + (" [follow-up: pulled extra context using prior question]" if is_followup else ""),
            retrieved=retrieved,
            answer=answer,
            backend=self.llm.backend,
            retrieval_backend=self.retrieval_backend,
            router_backend=self.router_backend,
            used_followup_memory=is_followup,
        )
        self._finalize(result, memory, session_id)
        return result

    def _answer_out_of_scope(self, question: str, history_block: str,
                              decision: RoutingDecision, reason: str) -> AssistantResult:
        """
        Handles any question that didn't match the local knowledge base.
        Tries live web search (Serper) first, if configured; falls back to
        the plain direct-LLM behavior (or its offline extractive summary)
        if web search is disabled, errors out, or finds nothing useful.
        """
        # Prefer Groq's server-side browser-search tool when a Groq key is
        # configured. This does not depend on the local machine reaching the
        # Serper endpoint. If Groq fails, retain Serper as a second fallback.
        if self.llm.groq_web_search_enabled:
            try:
                answer = self.llm.generate_from_groq_web_search(
                    system_prompt=SYSTEM_PROMPT_WEB,
                    user_prompt=build_direct_user_prompt(question, history_block),
                )
                return AssistantResult(
                    question=question,
                    route="web_search",
                    routing_confidence=decision.confidence,
                    routing_scores=decision.scores,
                    routing_reason=reason + " -> answered using Groq browser search.",
                    retrieved=[],
                    answer=answer,
                    backend="groq",
                    retrieval_backend=self.retrieval_backend,
                    router_backend=self.router_backend,
                )
            except Exception as exc:
                reason += f" -> Groq browser search failed ({type(exc).__name__}); trying Serper/direct_llm."

        if self.web_search.enabled:
            web_response = self.web_search.search(question)
            if web_response.ok:
                answer = self.llm.generate_from_web(
                    system_prompt=SYSTEM_PROMPT_WEB,
                    user_prompt=build_web_user_prompt(question, web_response, history_block),
                    web_response=web_response,
                )
                return AssistantResult(
                    question=question,
                    route="web_search",
                    routing_confidence=decision.confidence,
                    routing_scores=decision.scores,
                    routing_reason=reason + " -> answered from live web search (Serper).",
                    retrieved=[],
                    web_sources=[{"title": r.title, "link": r.link} for r in web_response.results],
                    answer=answer,
                    backend=self.llm.backend,
                    retrieval_backend=self.retrieval_backend,
                    router_backend=self.router_backend,
                )
            reason = reason + f" -> web search found nothing usable ({web_response.error}); falling back to direct_llm."

        answer = self.llm.generate(
            system_prompt=SYSTEM_PROMPT_DIRECT,
            user_prompt=build_direct_user_prompt(question, history_block),
            context_chunks=[],
        )
        return AssistantResult(
            question=question,
            route="direct_llm",
            routing_confidence=decision.confidence,
            routing_scores=decision.scores,
            routing_reason=reason,
            retrieved=[],
            answer=answer,
            backend=self.llm.backend,
            retrieval_backend=self.retrieval_backend,
            router_backend=self.router_backend,
        )

    def _finalize(self, result: AssistantResult, memory, session_id: str) -> None:
        memory.add(result.question, result.answer, result.route)
        if self.log_analytics:
            try:
                from . import analytics
                result.query_id = analytics.log_query(
                    question=result.question,
                    route=result.route,
                    confidence=result.routing_confidence,
                    backend=result.backend,
                    retrieval_backend=result.retrieval_backend,
                    citations=result.citations,
                    answer=result.answer,
                    session_id=session_id,
                )
            except Exception:
                # Analytics is best-effort and must never break the chat flow.
                pass
