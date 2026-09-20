
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from . import config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS queries (
    id TEXT PRIMARY KEY,
    ts TEXT NOT NULL,
    session_id TEXT,
    question TEXT NOT NULL,
    route TEXT,
    confidence REAL,
    backend TEXT,
    retrieval_backend TEXT,
    citations TEXT,
    answer_preview TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id TEXT PRIMARY KEY,
    query_id TEXT NOT NULL,
    ts TEXT NOT NULL,
    rating TEXT NOT NULL,   -- 'up' | 'down'
    comment TEXT,
    FOREIGN KEY (query_id) REFERENCES queries(id)
);

CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    messages TEXT NOT NULL   -- JSON-encoded list of chat message dicts
);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(str(config.ANALYTICS_DB_PATH))
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def log_query(question: str, route: str, confidence: float, backend: str,
              retrieval_backend: str, citations: List[str], answer: str,
              session_id: str = "default") -> str:
    init_db()
    query_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    preview = " ".join(answer.split())[:280]
    with _connect() as conn:
        conn.execute(
            "INSERT INTO queries (id, ts, session_id, question, route, confidence, "
            "backend, retrieval_backend, citations, answer_preview) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, ts, session_id, question, route, confidence, backend,
             retrieval_backend, "; ".join(citations), preview),
        )
    return query_id


def log_feedback(query_id: str, rating: str, comment: str = "") -> None:
    init_db()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO feedback (id, query_id, ts, rating, comment) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), query_id, datetime.now(timezone.utc).isoformat(), rating, comment),
        )


@dataclass
class Stats:
    total_queries: int
    total_sessions: int
    route_distribution: dict
    top_questions: List[tuple]
    activity_by_day: List[tuple]
    feedback_up: int
    feedback_down: int


def get_stats(top_n: int = 10) -> Stats:
    init_db()
    with _connect() as conn:
        total_queries = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        total_sessions = conn.execute("SELECT COUNT(DISTINCT session_id) FROM queries").fetchone()[0]

        route_rows = conn.execute(
            "SELECT COALESCE(route, 'unknown'), COUNT(*) FROM queries GROUP BY route"
        ).fetchall()
        route_distribution = {r: c for r, c in route_rows}

        top_questions = conn.execute(
            "SELECT question, COUNT(*) as n FROM queries GROUP BY LOWER(TRIM(question)) "
            "ORDER BY n DESC LIMIT ?", (top_n,)
        ).fetchall()

        activity_by_day = conn.execute(
            "SELECT SUBSTR(ts, 1, 10) as day, COUNT(*) FROM queries GROUP BY day ORDER BY day"
        ).fetchall()

        feedback_up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        feedback_down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]

    return Stats(
        total_queries=total_queries,
        total_sessions=total_sessions,
        route_distribution=route_distribution,
        top_questions=top_questions,
        activity_by_day=activity_by_day,
        feedback_up=feedback_up,
        feedback_down=feedback_down,
    )


def get_recent_queries(limit: int = 50):
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, session_id, question, route, confidence, backend, "
            "retrieval_backend, citations FROM queries ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return rows


def save_conversation(session_id: str, title: str, messages: list) -> None:
    """Upserts the full message history for a chat session, so it can be
    resumed later from the "Recent chats" sidebar list."""
    init_db()
    clean_title = " ".join((title or "New chat").split())[:80] or "New chat"
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (session_id, title, updated_at, messages) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "title=excluded.title, updated_at=excluded.updated_at, messages=excluded.messages",
            (session_id, clean_title, ts, json.dumps(messages)),
        )


def get_recent_conversations(limit: int = 15) -> List[dict]:
    """Returns recent chats (most-recently-updated first) for the sidebar
    history list: [{session_id, title, updated_at}, ...]."""
    init_db()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT session_id, title, updated_at FROM conversations "
            "ORDER BY updated_at DESC LIMIT ?", (limit,),
        ).fetchall()
    return [{"session_id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]


def load_conversation(session_id: str) -> Optional[list]:
    """Returns the stored message list for a session_id, or None if not found."""
    init_db()
    with _connect() as conn:
        row = conn.execute(
            "SELECT messages FROM conversations WHERE session_id = ?", (session_id,),
        ).fetchone()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except (TypeError, ValueError):
        return None


def delete_conversation(session_id: str) -> None:
    init_db()
    with _connect() as conn:
        conn.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))


def get_feedback_summary() -> Optional[dict]:
    init_db()
    with _connect() as conn:
        up = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='up'").fetchone()[0]
        down = conn.execute("SELECT COUNT(*) FROM feedback WHERE rating='down'").fetchone()[0]
        commented = conn.execute(
            "SELECT q.question, f.rating, f.comment, f.ts FROM feedback f "
            "JOIN queries q ON q.id = f.query_id WHERE TRIM(COALESCE(f.comment,'')) != '' "
            "ORDER BY f.ts DESC LIMIT 20"
        ).fetchall()
    total = up + down
    if total == 0:
        return None
    return {
        "up": up,
        "down": down,
        "total": total,
        "satisfaction_rate": up / total,
        "comments": commented,
    }
