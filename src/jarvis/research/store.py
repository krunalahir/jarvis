"""SQLite storage for research answers and content drafts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..schemas import PostDraft, ResearchAnswer

_SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS drafts (
    id TEXT PRIMARY KEY,
    platform TEXT NOT NULL,
    payload TEXT NOT NULL,
    scheduled_at TEXT,
    status TEXT NOT NULL
);
"""


class Store:
    def __init__(self, path: str = "jarvis_store.sqlite3"):
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # --- research answers ---

    def save_answer(self, answer: ResearchAnswer) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO answers (id, question, payload, created_at) VALUES (?, ?, ?, ?)",
            (answer.id, answer.question, answer.model_dump_json(), answer.created_at.isoformat()),
        )
        self._conn.commit()

    def get_answer(self, answer_id: str) -> Optional[ResearchAnswer]:
        row = self._conn.execute(
            "SELECT payload FROM answers WHERE id = ?", (answer_id,)
        ).fetchone()
        return ResearchAnswer(**json.loads(row[0])) if row else None

    def list_answers(self, limit: int = 50) -> list[ResearchAnswer]:
        rows = self._conn.execute(
            "SELECT payload FROM answers ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [ResearchAnswer(**json.loads(r[0])) for r in rows]

    # --- content drafts ---

    def save_draft(self, draft: PostDraft) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO drafts (id, platform, payload, scheduled_at, status) VALUES (?, ?, ?, ?, ?)",
            (
                draft.id,
                draft.platform.value,
                draft.model_dump_json(),
                draft.scheduled_at.isoformat() if draft.scheduled_at else None,
                draft.status.value,
            ),
        )
        self._conn.commit()

    def get_draft(self, draft_id: str) -> Optional[PostDraft]:
        row = self._conn.execute(
            "SELECT payload FROM drafts WHERE id = ?", (draft_id,)
        ).fetchone()
        return PostDraft(**json.loads(row[0])) if row else None

    def due_drafts(self, now_iso: str) -> list[PostDraft]:
        rows = self._conn.execute(
            "SELECT payload FROM drafts WHERE status = 'scheduled' AND scheduled_at <= ?",
            (now_iso,),
        ).fetchall()
        return [PostDraft(**json.loads(r[0])) for r in rows]
