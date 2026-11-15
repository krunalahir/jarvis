"""Social platform connectors.

What each platform actually allows (checked against official docs,
November 2026 - re-verify before relying on these):

- Instagram: Content Publishing API for professional (business/creator)
  accounts. Two-step: create a media container, then publish it. Media
  must be at a public URL. 100 API-published posts per 24h. The API
  publishes immediately; scheduling is the app's job (see scheduler.py).
- X: posting works, pay-per-use, no permanent free tier.
- YouTube: videos.insert supports true native scheduling: upload as
  private with status.publishAt set; YouTube releases it at that time.
  Documented upload quota: 100 calls/day in the Video Uploads bucket.
- Snapchat: no general public posting API. Snap's Public Profile API
  (partner access) can post Stories/Spotlights for Public Profiles;
  Creative Kit only covers user-initiated sharing. Included here as a
  scaffold that fails loudly with an explanation.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Optional

from ..schemas import PostDraft
from .base import Connector, PlatformAccessRequired


class InstagramConnector(Connector):
    name = "instagram"
    required_settings = ("meta_access_token", "instagram_business_account_id")
    graph = "https://graph.facebook.com/v21.0"

    def _create_container(self, media_url: str, caption: str, is_video: bool) -> str:
        ig_id = self.settings.instagram_business_account_id
        payload = {"caption": caption, "access_token": self.settings.meta_access_token}
        if is_video:
            payload["media_type"] = "REELS"
            payload["video_url"] = media_url
        else:
            payload["image_url"] = media_url
        response = self._client.post(f"{self.graph}/{ig_id}/media", data=payload)
        response.raise_for_status()
        return response.json()["id"]

    def publish_draft(self, draft: PostDraft) -> dict:
        """Publish a draft immediately. Media must be at a public URL.
        Write operation: gated on credentials and actions enabled."""
        self.require_enabled()
        if not draft.media_urls:
            raise ValueError("Instagram requires at least one media URL")
        container_id = self._create_container(
            draft.media_urls[0], draft.text, is_video=draft.media_urls[0].endswith((".mp4", ".mov"))
        )
        ig_id = self.settings.instagram_business_account_id
        response = self._client.post(
            f"{self.graph}/{ig_id}/media_publish",
            data={
                "creation_id": container_id,
                "access_token": self.settings.meta_access_token,
            },
        )
        response.raise_for_status()
        return response.json()


class XConnector(Connector):
    name = "x"
    required_settings = ("x_bearer_token",)
    api = "https://api.x.com/2/tweets"

    def publish_draft(self, draft: PostDraft) -> dict:
        """Post a tweet. X API access is pay-per-use; every call costs
        money on the connected X developer account. Gated write."""
        self.require_enabled()
        response = self._client.post(
            self.api,
            headers={"Authorization": f"Bearer {self.settings.x_bearer_token}"},
            json={"text": draft.text},
        )
        response.raise_for_status()
        return response.json()


class YouTubeConnector(Connector):
    name = "youtube"
    required_settings = ("google_access_token",)
    api = "https://www.googleapis.com/upload/youtube/v3/videos"

    def publish_draft(self, draft: PostDraft, video_file_path: Optional[str] = None) -> dict:
        """Upload a video; schedules natively when draft.scheduled_at is set
        (uploaded private, released at publishAt). Gated write.

        The video comes from video_file_path, or is downloaded from the
        draft's first media URL when no path is given - drafts created
        through the API or scheduler carry media URLs, not local paths.

        Uses the single-request multipart upload; fine for short clips.
        For large files switch to the resumable upload protocol.
        """
        self.require_enabled()
        temp_path: Optional[str] = None
        if not video_file_path:
            if not draft.media_urls:
                raise ValueError(
                    "YouTube publishing needs a video file path or a media URL on the draft"
                )
            temp_path = self._download(draft.media_urls[0])
            video_file_path = temp_path
        status = {"privacyStatus": "private"}
        if draft.scheduled_at:
            status["publishAt"] = draft.scheduled_at.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        metadata = {
            "snippet": {"title": draft.text[:100], "description": draft.text},
            "status": status,
        }
        try:
            with open(video_file_path, "rb") as handle:
                response = self._client.post(
                    self.api,
                    params={"part": "snippet,status", "uploadType": "multipart"},
                    headers={"Authorization": f"Bearer {self.settings.google_access_token}"},
                    files={
                        "metadata": ("metadata.json", json.dumps(metadata), "application/json"),
                        "video": ("video.mp4", handle, "video/mp4"),
                    },
                )
            response.raise_for_status()
            return response.json()
        finally:
            if temp_path:
                os.unlink(temp_path)

    def _download(self, url: str) -> str:
        """Fetch a public media URL to a temp file and return its path."""
        with self._client.stream("GET", url) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                for chunk in response.iter_bytes():
                    tmp.write(chunk)
                return tmp.name


class SnapchatConnector(Connector):
    """Scaffold only. Posting to Snapchat requires Snap's Public Profile
    API, which is partner-gated and serves brands/creators with Public
    Profiles - not personal accounts. Apply for partner access first.

    Also note: the platform name 'snaptrack' from the original request was
    read as Snapchat but never confirmed."""

    name = "snapchat"
    required_settings: tuple[str, ...] = ()

    def publish_draft(self, draft: PostDraft) -> dict:
        raise PlatformAccessRequired(
            "Snapchat posting needs Snap Public Profile API partner access. "
            "Creative Kit only supports user-initiated sharing from the app."
        )

