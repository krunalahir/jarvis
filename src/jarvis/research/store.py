"""SQLite storage for research answers and content drafts."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Optional

from ..schemas import (
    ConversationMemory,
    DraftStatus,
    PendingAction,
    PostDraft,
    ResearchAnswer,
    ResearchJob,
)

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
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS actions (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    updated_at TEXT NOT NULL
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

    def claim_due_drafts(self, now_iso: str) -> list[PostDraft]:
        """Atomically move due drafts scheduled -> publishing and return them.

        Claiming first means two overlapping publish runs can never pick up
        the same draft, so a draft cannot be posted twice.
        """
        with self._conn:
            ids = [
                r[0]
                for r in self._conn.execute(
                    "SELECT id FROM drafts WHERE status = 'scheduled' AND scheduled_at <= ?",
                    (now_iso,),
                ).fetchall()
            ]
            claimed = []
            for draft_id in ids:
                cur = self._conn.execute(
                    "UPDATE drafts SET status = 'publishing' WHERE id = ? AND status = 'scheduled'",
                    (draft_id,),
                )
                if cur.rowcount == 1:
                    row = self._conn.execute(
                        "SELECT payload FROM drafts WHERE id = ?", (draft_id,)
                    ).fetchone()
                    draft = PostDraft(**json.loads(row[0]))
                    draft.status = DraftStatus.PUBLISHING
                    self._conn.execute(
                        "UPDATE drafts SET payload = ? WHERE id = ?",
                        (draft.model_dump_json(), draft_id),
                    )
                    claimed.append(draft)
        return claimed

    # --- research jobs ---

    def save_job(self, job: ResearchJob) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO jobs (id, question, payload, status) VALUES (?, ?, ?, ?)",
            (job.id, job.question, job.model_dump_json(), job.status.value),
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        row = self._conn.execute(
            "SELECT payload FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return ResearchJob(**json.loads(row[0])) if row else None

    def list_jobs(self, limit: int = 50) -> list[ResearchJob]:
        rows = self._conn.execute(
            "SELECT payload FROM jobs ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [ResearchJob(**json.loads(r[0])) for r in rows]

    def interrupted_jobs(self) -> list[ResearchJob]:
        rows = self._conn.execute(
            "SELECT payload FROM jobs WHERE status IN ('pending', 'running')"
        ).fetchall()
        return [ResearchJob(**json.loads(r[0])) for r in rows]

    # --- pending actions ---

    def save_action(self, action: PendingAction) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO actions (id, kind, payload, status) VALUES (?, ?, ?, ?)",
            (action.id, action.kind.value, action.model_dump_json(), action.status.value),
        )
        self._conn.commit()

    def get_action(self, action_id: str) -> Optional[PendingAction]:
        row = self._conn.execute(
            "SELECT payload FROM actions WHERE id = ?", (action_id,)
        ).fetchone()
        return PendingAction(**json.loads(row[0])) if row else None

    def list_actions(self, limit: int = 50) -> list[PendingAction]:
        rows = self._conn.execute(
            "SELECT payload FROM actions ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [PendingAction(**json.loads(r[0])) for r in rows]

    # --- conversation memory ---

    def save_conversation(self, memory: ConversationMemory) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO conversations (id, payload, updated_at) VALUES (?, ?, ?)",
            (memory.id, memory.model_dump_json(), memory.updated_at.isoformat()),
        )
        self._conn.commit()

    def get_conversation(self, conversation_id: str) -> Optional[ConversationMemory]:
        row = self._conn.execute(
            "SELECT payload FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        return ConversationMemory(**json.loads(row[0])) if row else None
