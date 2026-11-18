"""The task monitor: JARVIS watches every task from start to finish.

The router (agent/router.py) decides what a request is; the agent loop
(agent/graph.py) does the work. The monitor wraps the whole run and
reports on it like a supervisor: it records when the request arrives,
which route was picked, whether the needed app is authenticated, each
tool the agent uses, every approval the user still has to give, any
recovery attempt after a failure, and the final outcome.

Each event is a TaskUpdate written for the chat, so the user sees
progress while the task runs instead of a silent spinner. Updates are
kept on the monitor for the live reply and persisted in the store, so a
client can also poll GET /agent/tasks/{task_id}/updates after the fact.

Recovery policy: a failed run is retried once. Reads are safe to re-run
and writes never execute on their own (the approval gate queues them),
so a retry cannot double-send or double-post. A run that fails again
stops and says so; nothing loops forever.
"""

from __future__ import annotations

import logging
from typing import Optional

from ..connectors.base import Connector
from ..research.store import Store
from ..schemas import AgentReply, AgentRunStatus, TaskStage, TaskUpdate, new_id

log = logging.getLogger(__name__)


class TaskMonitor:
    """Supervises one task run and records its progress trail."""

    # One initial attempt plus one recovery retry.
    MAX_ATTEMPTS = 2

    def __init__(self, store: Optional[Store] = None, task_id: Optional[str] = None):
        self.task_id = task_id or new_id()
        self._store = store
        self._updates: list[TaskUpdate] = []

    @property
    def updates(self) -> list[TaskUpdate]:
        return list(self._updates)

    def update(self, stage: TaskStage, message: str) -> TaskUpdate:
        update = TaskUpdate(
            task_id=self.task_id,
            seq=len(self._updates) + 1,
            stage=stage,
            message=message,
        )
        self._updates.append(update)
        if self._store is not None:
            self._store.save_task_update(update)
        return update

    def check_auth(self, connector: Connector) -> bool:
        """Record whether the app this task needs is ready.

        Authentication itself is automatic: connectors run on their saved
        credentials with no user step in the loop. What the monitor adds
        is visibility - it says up front when a credential is missing so
        the chat shows exactly what to configure instead of a vague
        failure later. Returns True when the connector is fully ready.
        """
        status = connector.status()
        state = status["state"]
        if state == "ready":
            self.update(TaskStage.AUTH_CHECK, f"{connector.name}: connected and ready.")
            return True
        if state == "not_configured":
            # Settings fields are lowercase; show the env vars the user
            # actually sets (.env.example naming), e.g. GOOGLE_ACCESS_TOKEN.
            missing = ", ".join(m.upper() for m in status["missing_settings"]) or "credentials"
            self.update(
                TaskStage.AUTH_CHECK,
                f"{connector.name}: not connected yet - missing {missing}. "
                "Once it is configured, JARVIS authenticates automatically; "
                "no sign-in step is needed during a task.",
            )
        else:
            self.update(
                TaskStage.AUTH_CHECK,
                f"{connector.name}: credentials are configured but actions are "
                "disabled (JARVIS_ACTIONS_ENABLED is false), so writes stay "
                "queued for approval.",
            )
        return False

    def should_retry(self, reply: AgentReply, attempt: int) -> bool:
        """True when a failed attempt may be tried once more.

        `attempt` is zero-based: attempt 0 is the first run. Only FAILED
        runs are retried; a stopped-at-step-limit run is a deliberate
        stop, not a transient error.
        """
        return reply.status == AgentRunStatus.FAILED and attempt + 1 < self.MAX_ATTEMPTS
