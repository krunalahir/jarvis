"""Per-action approval gate.

JARVIS_ACTIONS_ENABLED gates capability. It is NOT consent. Every
real-world action (send, post, call, play, publish) becomes a
PendingAction first and executes only after the user approves that
specific action through POST /actions/{id}/approve.
"""

from __future__ import annotations

from typing import Callable, Optional

from .research.store import Store
from .schemas import ActionKind, PendingAction, PendingActionStatus, utcnow

Executor = Callable[[dict], dict]


class ActionGate:
    def __init__(self, store: Store, executors: dict[ActionKind, Executor]):
        self.store = store
        self._executors = executors

    def request(self, kind: ActionKind, summary: str, payload: dict) -> PendingAction:
        action = PendingAction(kind=kind, summary=summary, payload=payload)
        self.store.save_action(action)
        return action

    def get(self, action_id: str) -> Optional[PendingAction]:
        return self.store.get_action(action_id)

    def list(self, limit: int = 50) -> list[PendingAction]:
        return self.store.list_actions(limit)

    def approve(self, action_id: str) -> PendingAction:
        action = self.store.get_action(action_id)
        if action is None:
            raise KeyError(f"unknown action: {action_id}")
        if action.status != PendingActionStatus.PENDING:
            raise ValueError(f"action {action_id} is already {action.status.value}")
        executor = self._executors.get(action.kind)
        try:
            if executor is None:
                raise ValueError(f"no executor for {action.kind.value}")
            result = executor(action.payload)
            action.status = PendingActionStatus.EXECUTED
            action.result = str(result)[:500]
            action.error = None
        except Exception as exc:  # visible failure stays on the action
            action.status = PendingActionStatus.FAILED
            action.error = str(exc)
        action.decided_at = utcnow()
        self.store.save_action(action)
        return action

    def reject(self, action_id: str) -> PendingAction:
        action = self.store.get_action(action_id)
        if action is None:
            raise KeyError(f"unknown action: {action_id}")
        if action.status != PendingActionStatus.PENDING:
            raise ValueError(f"action {action_id} is already {action.status.value}")
        action.status = PendingActionStatus.REJECTED
        action.decided_at = utcnow()
        self.store.save_action(action)
        return action
