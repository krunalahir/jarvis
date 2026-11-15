"""Crash recovery for interrupted real-world effects.

A restart can find actions stuck in 'executing' or drafts stuck in
'publishing'. In both cases the external effect may ALREADY have
happened (the email may be sent, the post may be live) and the process
died before it could record the result. Blindly retrying would risk
sending or posting twice, so recovery never retries: it marks each item
failed with an unknown-outcome note, and a human checks the platform
before deciding to run it again.

Research jobs are different: re-running one has no external side effect,
so JobRunner requeues those on startup (see research/jobs.py).
"""

from __future__ import annotations

from .research.store import Store
from .schemas import DraftStatus, PendingActionStatus, utcnow

UNKNOWN_OUTCOME = (
    "Interrupted by a restart while in progress. The external action may "
    "already have happened. It was NOT retried automatically - check the "
    "service first, then re-request the action if it did not go through."
)


def recover_interrupted_state(store: Store) -> dict:
    """Mark interrupted actions and drafts as failed-unknown. No retries."""
    actions = store.interrupted_actions()
    for action in actions:
        action.status = PendingActionStatus.FAILED
        action.error = UNKNOWN_OUTCOME
        action.decided_at = utcnow()
        store.save_action(action)

    drafts = store.interrupted_drafts()
    for draft in drafts:
        draft.status = DraftStatus.FAILED
        draft.error = UNKNOWN_OUTCOME
        store.save_draft(draft)

    return {"actions": len(actions), "drafts": len(drafts)}
