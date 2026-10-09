import os
os.environ["LLM_BACKEND"] = "extractive"

import pytest

from src.assistant import TrainingAssistant


@pytest.fixture(scope="module")
def assistant():
    return TrainingAssistant()


def test_corpus_loaded(assistant):
    assert len(assistant.chunks) > 10


@pytest.mark.parametrize("question,expected_route", [
    ("What are the company's core values?", "general_company"),
    ("What are standard work hours?", "general_company"),
    ("What are a Data Analyst's first 30 days expectations?", "role_specific"),
    ("Who should a Product Manager ask for feasibility checks?", "role_specific"),
    ("How do I submit an expense claim?", "admin_policy"),
    ("How do I request PTO?", "admin_policy"),
    ("What is my exact salary breakup?", "direct_llm"),
    ("Can you approve my leave request right now?", "direct_llm"),
])
def test_routing(assistant, question, expected_route):
    decision = assistant.route_only(question)
    assert decision.route == expected_route, (
        f"Q: {question!r} -> got {decision.route}, scores={decision.scores}"
    )


def test_rag_answer_has_citation(assistant):
    result = assistant.ask("How do I submit an expense claim and by when?")
    assert result.route == "admin_policy"
    assert result.citations, "Expected at least one citation for a policy question"
    assert any("expense_policy" in c for c in result.citations)


def test_direct_llm_has_no_citation(assistant):
    result = assistant.ask("Can you approve my leave request right now?")
    assert result.route == "direct_llm"
    assert result.citations == []


def test_answer_not_empty(assistant):
    result = assistant.ask("What tools does a Data Analyst commonly use?")
    assert result.answer.strip() != ""
