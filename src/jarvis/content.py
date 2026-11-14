"""Content creation: turn a topic into a platform-ready post draft.

Uses OpenAI when OPENAI_API_KEY is set. Without it, generation is
unavailable - JARVIS does not pretend to write content it cannot.
Generating a draft never publishes anything; publishing is a separate,
approved action.
"""

from __future__ import annotations

from typing import Optional

import httpx

from .config import Settings
from .schemas import Platform


class ContentGenerationUnavailable(Exception):
    pass


_LENGTH_HINTS = {
    Platform.X: "at most 280 characters",
    Platform.INSTAGRAM: "a caption of a few short paragraphs",
    Platform.YOUTUBE: "a title (max 100 chars) on the first line, then a description",
    Platform.SNAPCHAT: "a short caption",
}


class ContentGenerator:
    endpoint = "https://api.openai.com/v1/chat/completions"

    def __init__(self, settings: Settings, client: Optional[httpx.Client] = None):
        self._api_key = settings.openai_api_key
        self._client = client or httpx.Client(timeout=60)

    def available(self) -> bool:
        return bool(self._api_key)

    def generate(self, platform: Platform, topic: str, tone: Optional[str] = None) -> str:
        if not self.available():
            raise ContentGenerationUnavailable(
                "Content generation needs OPENAI_API_KEY. Set it, or write the "
                "post text yourself and create the draft directly."
            )
        length = _LENGTH_HINTS.get(platform, "a short post")
        prompt = (
            f"Write {length} for {platform.value} about: {topic}. "
            f"Tone: {tone or 'clear and direct'}. Return only the post text."
        )
        response = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": "gpt-4.1-mini",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
