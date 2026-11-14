"""Scheduled publishing.

Drafts are created first and stay inert. Scheduling sets a time; the
scheduler publishes due drafts through the right connector. Publishing
still requires that connector to be configured and actions enabled -
a due draft with no credentials fails visibly, never silently.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .schemas import DraftStatus, Platform, PostDraft, utcnow
from .research.store import Store
from .connectors.base import ConnectorError
from .connectors.social import InstagramConnector, SnapchatConnector, XConnector, YouTubeConnector

PUBLISHERS = {
    Platform.INSTAGRAM: InstagramConnector,
    Platform.X: XConnector,
    Platform.YOUTUBE: YouTubeConnector,
    Platform.SNAPCHAT: SnapchatConnector,
}


class PublishScheduler:
    def __init__(self, store: Store, connectors: dict):
        self.store = store
        self.connectors = connectors

    def create_draft(self, platform: Platform, text: str, media_urls: list[str] | None = None) -> PostDraft:
        draft = PostDraft(platform=platform, text=text, media_urls=media_urls or [])
        self.store.save_draft(draft)
        return draft

    def schedule(self, draft_id: str, run_at: datetime) -> PostDraft:
        draft = self.store.get_draft(draft_id)
        if draft is None:
            raise KeyError(f"unknown draft: {draft_id}")
        if run_at.tzinfo is None:
            run_at = run_at.replace(tzinfo=timezone.utc)
        draft.scheduled_at = run_at
        draft.status = DraftStatus.SCHEDULED
        self.store.save_draft(draft)
        return draft

    def publish_due(self) -> list[PostDraft]:
        """Publish every scheduled draft whose time has passed. Claims drafts
        atomically first, so overlapping runs can never post the same draft
        twice. Returns the drafts it touched, PUBLISHED or FAILED on each."""
        now = utcnow().isoformat()
        touched = []
        for draft in self.store.claim_due_drafts(now):
            if draft.platform_post_id:
                # Already has a platform id: never post it again.
                draft.status = DraftStatus.PUBLISHED
                self.store.save_draft(draft)
                continue
            connector_cls = PUBLISHERS.get(draft.platform)
            connector = self.connectors.get(connector_cls.name) if connector_cls else None
            try:
                if connector is None:
                    raise ConnectorError(f"no publisher for {draft.platform}")
                result = connector.publish_draft(draft)
                draft.status = DraftStatus.PUBLISHED
                draft.platform_post_id = str(result.get("id", "")) or None
                draft.error = None
            except Exception as exc:  # visible failure, kept on the draft
                draft.status = DraftStatus.FAILED
                draft.error = str(exc)
            self.store.save_draft(draft)
            touched.append(draft)
        return touched
