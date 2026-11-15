"""SQLite storage for research answers and content drafts.

One Store is shared by the API worker threads, the background job pool
and the scheduler. SQLite serializes drivers are not enough on their own
here because several operations read a row, decide, then write it back,
so every method takes a process-wide re-entrant lock and the connection
runs with WAL and a busy timeout.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from ..schemas import (
    DraftStatus,
    PendingAction,
    PendingActionStatus,
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
"""


class Store:
    def __init__(self, path: str = "jarvis_store.sqlite3"):
        self._path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, check_same_thread=False)
        # Wait on a briefly-locked database instead of erroring, and use
        # WAL so readers do not block writers. Harmless on :memory:.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.executescript(_SCHEMA)
        # One lock for the whole store: methods do read-modify-write, so
        # per-statement SQLite serialization alone is not sufficient.
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- research answers ---

    def save_answer(self, answer: ResearchAnswer) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO answers (id, question, payload, created_at) VALUES (?, ?, ?, ?)",
                (answer.id, answer.question, answer.model_dump_json(), answer.created_at.isoformat()),
            )
            self._conn.commit()

    def get_answer(self, answer_id: str) -> Optional[ResearchAnswer]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM answers WHERE id = ?", (answer_id,)
            ).fetchone()
        return ResearchAnswer(**json.loads(row[0])) if row else None

    def list_answers(self, limit: int = 50) -> list[ResearchAnswer]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM answers ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [ResearchAnswer(**json.loads(r[0])) for r in rows]

    # --- content drafts ---

    def save_draft(self, draft: PostDraft) -> None:
        with self._lock:
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
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM drafts WHERE id = ?", (draft_id,)
            ).fetchone()
        return PostDraft(**json.loads(row[0])) if row else None

    def due_drafts(self, now_iso: str) -> list[PostDraft]:
        with self._lock:
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
        with self._lock, self._conn:
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

    def interrupted_drafts(self) -> list[PostDraft]:
        """Drafts left in 'publishing' by a crash or restart."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM drafts WHERE status = 'publishing'"
            ).fetchall()
        return [PostDraft(**json.loads(r[0])) for r in rows]

    # --- research jobs ---

    def save_job(self, job: ResearchJob) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs (id, question, payload, status) VALUES (?, ?, ?, ?)",
                (job.id, job.question, job.model_dump_json(), job.status.value),
            )
            self._conn.commit()

    def get_job(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
        return ResearchJob(**json.loads(row[0])) if row else None

    def list_jobs(self, limit: int = 50) -> list[ResearchJob]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM jobs ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [ResearchJob(**json.loads(r[0])) for r in rows]

    def interrupted_jobs(self) -> list[ResearchJob]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM jobs WHERE status IN ('pending', 'running')"
            ).fetchall()
        return [ResearchJob(**json.loads(r[0])) for r in rows]

    # --- pending actions ---

    def save_action(self, action: PendingAction) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO actions (id, kind, payload, status) VALUES (?, ?, ?, ?)",
                (action.id, action.kind.value, action.model_dump_json(), action.status.value),
            )
            self._conn.commit()

    def get_action(self, action_id: str) -> Optional[PendingAction]:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM actions WHERE id = ?", (action_id,)
            ).fetchone()
        return PendingAction(**json.loads(row[0])) if row else None

    def list_actions(self, limit: int = 50) -> list[PendingAction]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM actions ORDER BY json_extract(payload, '$.created_at') DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [PendingAction(**json.loads(r[0])) for r in rows]

    def claim_action(self, action_id: str) -> Optional[PendingAction]:
        """Atomically move an action pending -> executing and return it.

        Exactly one caller can win the claim, so concurrent approve
        requests (double click, retry, two clients) can never run the
        same real-world action twice. A crash after the claim leaves the
        action in 'executing' for interrupted_actions() to surface.
        """
        return self._transition_action(
            action_id,
            from_status=PendingActionStatus.PENDING,
            to_status=PendingActionStatus.EXECUTING,
        )

    def reject_action(self, action_id: str) -> Optional[PendingAction]:
        """Atomically move an action pending -> rejected and return it."""
        return self._transition_action(
            action_id,
            from_status=PendingActionStatus.PENDING,
            to_status=PendingActionStatus.REJECTED,
        )

    def _transition_action(
        self,
        action_id: str,
        from_status: PendingActionStatus,
        to_status: PendingActionStatus,
    ) -> Optional[PendingAction]:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "UPDATE actions SET status = ? WHERE id = ? AND status = ?",
                (to_status.value, action_id, from_status.value),
            )
            if cur.rowcount != 1:
                return None
            row = self._conn.execute(
                "SELECT payload FROM actions WHERE id = ?", (action_id,)
            ).fetchone()
            action = PendingAction(**json.loads(row[0]))
            action.status = to_status
            self._conn.execute(
                "UPDATE actions SET payload = ? WHERE id = ?",
                (action.model_dump_json(), action_id),
            )
        return action

    def interrupted_actions(self) -> list[PendingAction]:
        """Actions left in 'executing' by a crash or restart."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload FROM actions WHERE status = 'executing'"
            ).fetchall()
        return [PendingAction(**json.loads(r[0])) for r in rows]
