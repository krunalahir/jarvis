"""Spotify playback through the Web API.

Startup note: new Spotify apps run in development mode - capped at 5
allowlisted users, and the account owner needs Premium for playback.
Apply for extended quota mode before a real launch.
"""

from __future__ import annotations

from typing import Optional

from .base import Connector


class SpotifyConnector(Connector):
    name = "spotify"
    required_settings = ("spotify_access_token",)
    api = "https://api.spotify.com/v1"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.spotify_access_token}"}

    def search_track(self, query: str, limit: int = 5) -> list[dict]:
        """Read operation: only needs configuration, not the actions switch."""
        self.require_configured()
        response = self._client.get(
            f"{self.api}/search",
            headers=self._headers(),
            params={"q": query, "type": "track", "limit": limit},
        )
        response.raise_for_status()
        return [
            {
                "uri": t["uri"],
                "name": t["name"],
                "artists": [a["name"] for a in t["artists"]],
            }
            for t in response.json()["tracks"]["items"]
        ]

    def play(self, track_query: Optional[str] = None, device_id: Optional[str] = None) -> dict:
        """Start playback. With a query, plays the top matching track.
        Write operation: gated on credentials and actions enabled."""
        self.require_enabled()
        body: dict = {}
        if track_query:
            matches = self.search_track(track_query, limit=1)
            if not matches:
                raise ValueError(f"No Spotify track found for: {track_query!r}")
            body["uris"] = [matches[0]["uri"]]
        params = {"device_id": device_id} if device_id else {}
        response = self._client.put(
            f"{self.api}/me/player/play",
            headers=self._headers(),
            params=params,
            json=body,
        )
        # 204 = playback started; 200/202 also acceptable per device state
        if response.status_code not in (200, 202, 204):
            response.raise_for_status()
        return {"started": True, "status_code": response.status_code, "body": body}
