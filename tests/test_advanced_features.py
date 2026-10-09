import os

os.environ["LLM_BACKEND"] = "extractive"

import tempfile

import pytest

from src.assistant import TrainingAssistant
from src.memory import ConversationMemory, SessionMemoryStore


# ---------------------------------------------------------------------------
# FAISS vector retrieval backend
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def faiss_assistant():
    return TrainingAssistant(retrieval_backend="faiss", log_analytics=False)


def test_faiss_backend_selected(faiss_assistant):
    assert faiss_assistant.retrieval_backend == "faiss"


def test_faiss_retrieval_returns_citations(faiss_assistant):
    result = faiss_assistant.ask("How do I submit an expense claim?", session_id="faiss-test")
    assert result.route == "admin_policy"
    assert result.citations, "FAISS retriever should still surface citations"
    assert any("expense_policy" in c for c in result.citations)


def test_faiss_direct_llm_still_works(faiss_assistant):
    result = faiss_assistant.ask("Can you approve my leave request right now?", session_id="faiss-test-2")
    assert result.route == "direct_llm"
    assert result.citations == []


# ---------------------------------------------------------------------------
# LLM-based router (falls back to rule-based in offline/extractive mode)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def llm_router_assistant():
    return TrainingAssistant(router_backend="llm", log_analytics=False)


def test_llm_router_falls_back_when_offline(llm_router_assistant):
    decision = llm_router_assistant.route_only("How do I submit an expense claim?")
    assert decision.route == "admin_policy"
    assert "rule_based" in decision.reason or "llm" in decision.reason.lower()


def test_llm_router_sensitive_override_still_applies(llm_router_assistant):
    decision = llm_router_assistant.route_only("What is my exact salary breakup?")
    assert decision.route == "direct_llm"


# ---------------------------------------------------------------------------
# Conversation memory
# ---------------------------------------------------------------------------

def test_conversation_memory_records_turns():
    mem = ConversationMemory(max_turns=3)
    assert mem.is_empty()
    mem.add("Q1", "A1", "admin_policy")
    mem.add("Q2", "A2", "role_specific")
    assert len(mem.turns) == 2
    assert not mem.is_empty()
    block = mem.as_prompt_block()
    assert "Q1" in block and "A1" in block


def test_conversation_memory_bounded():
    mem = ConversationMemory(max_turns=2)
    mem.add("Q1", "A1")
    mem.add("Q2", "A2")
    mem.add("Q3", "A3")
    assert len(mem.turns) == 2
    questions = [t.question for t in mem.turns]
    assert "Q1" not in questions  # oldest turn evicted


def test_session_memory_store_isolates_sessions():
    store = SessionMemoryStore(max_turns=4)
    store.get("session-a").add("A-Q1", "A-A1")
    store.get("session-b").add("B-Q1", "B-A1")
    assert len(store.get("session-a").turns) == 1
    assert len(store.get("session-b").turns) == 1
    assert store.get("session-a").turns[0].question == "A-Q1"


def test_assistant_uses_conversation_memory_across_turns():
    assistant = TrainingAssistant(log_analytics=False)
    assistant.ask("What are a Data Analyst's first 30 days expectations?", session_id="mem-test")
    memory = assistant.memory.get("mem-test")
    assert len(memory.turns) == 1
    assistant.ask("What about for a Product Manager instead?", session_id="mem-test")
    assert len(memory.turns) == 2


def test_reset_conversation_clears_memory():
    assistant = TrainingAssistant(log_analytics=False)
    assistant.ask("How do I submit an expense claim?", session_id="reset-test")
    assert not assistant.memory.get("reset-test").is_empty()
    assistant.reset_conversation("reset-test")
    assert assistant.memory.get("reset-test").is_empty()


# ---------------------------------------------------------------------------
# Analytics store
# ---------------------------------------------------------------------------

def test_analytics_logs_and_reads_queries(monkeypatch, tmp_path):
    from src import analytics, config

    monkeypatch.setattr(config, "ANALYTICS_DB_PATH", tmp_path / "test_analytics.db")

    query_id = analytics.log_query(
        question="Test question?",
        route="admin_policy",
        confidence=0.5,
        backend="extractive",
        retrieval_backend="tfidf",
        citations=["policy.md#Section"],
        answer="Test answer.",
        session_id="test-session",
    )
    assert query_id

    analytics.log_feedback(query_id, "up", "great answer")

    stats = analytics.get_stats()
    assert stats.total_queries == 1
    assert stats.feedback_up == 1
    assert stats.feedback_down == 0

    summary = analytics.get_feedback_summary()
    assert summary is not None
    assert summary["up"] == 1
    assert summary["total"] == 1
