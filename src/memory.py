
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional


@dataclass
class Turn:
    question: str
    answer: str
    route: str = ""


class ConversationMemory:
    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self._turns: Deque[Turn] = deque(maxlen=max_turns)

    def add(self, question: str, answer: str, route: str = "") -> None:
        self._turns.append(Turn(question=question, answer=answer, route=route))

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()

    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def as_prompt_block(self) -> str:
        """Render prior turns as a compact text block for prompt context.
        Answers are truncated so the memory window can't balloon the
        prompt size as a conversation gets long."""
        if not self._turns:
            return ""
        lines = []
        for t in self._turns:
            short_answer = " ".join(t.answer.split())
            if len(short_answer) > 220:
                short_answer = short_answer[:217].rstrip() + "..."
            lines.append(f"Employee asked: {t.question}\nAssistant answered: {short_answer}")
        return "\n\n".join(lines)

    def rewrite_query_hint(self) -> Optional[str]:
        """Return the single most recent question, useful as light
        context when a follow-up question is short/ambiguous (e.g. just
        'what about for a PM?')."""
        if not self._turns:
            return None
        return self._turns[-1].question


class SessionMemoryStore:
    """In-process registry mapping session_id -> ConversationMemory, used
    by the Streamlit app (one browser session) and the CLI (one process)."""

    def __init__(self, max_turns: int = 4):
        self.max_turns = max_turns
        self._sessions = {}

    def get(self, session_id: str) -> ConversationMemory:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationMemory(max_turns=self.max_turns)
        return self._sessions[session_id]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
