"""Google connectors: Gmail and Calendar.

One Google OAuth access token covers both, plus YouTube (see social.py).
Required scopes: gmail.send (plus gmail.readonly for reading mail),
calendar.events, youtube.upload.
JARVIS never stores your Google credentials; supply a token via
GOOGLE_ACCESS_TOKEN from your own OAuth flow.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Optional

from ..schemas import utcnow
from .base import Connector


class GmailConnector(Connector):
    name = "gmail"
    required_settings = ("google_access_token",)
    api = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.google_access_token}"}

    def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        """Read operation: needs configuration, not the actions switch."""
        self.require_configured()
        response = self._client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            headers=self._headers(),
            params={"q": query, "maxResults": limit},
        )
        response.raise_for_status()
        return response.json().get("messages", [])

    def send_email(self, to: str, subject: str, body: str, cc: Optional[list[str]] = None) -> dict:
        """Send an email as the connected Google account. This is a write:
        it requires credentials and JARVIS_ACTIONS_ENABLED=true."""
        self.require_enabled()
        message = EmailMessage()
        message["To"] = to
        if cc:
            message["Cc"] = ", ".join(cc)
        message["Subject"] = subject
        message.set_content(body)
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        response = self._client.post(
            self.api,
            headers=self._headers(),
            json={"raw": raw},
        )
        response.raise_for_status()
        return response.json()


class CalendarConnector(Connector):
    name = "google_calendar"
    required_settings = ("google_access_token",)
    api = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.settings.google_access_token}"}

    def list_events(self, time_min_iso: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Read operation: needs configuration, not the actions switch."""
        self.require_configured()
        params: dict = {"maxResults": limit, "singleEvents": "true", "orderBy": "startTime"}
        if time_min_iso:
            params["timeMin"] = time_min_iso
        response = self._client.get(self.api, headers=self._headers(), params=params)
        response.raise_for_status()
        return response.json().get("items", [])

    def create_event(
        self,
        summary: str,
        start_iso: str,
        end_iso: str,
        attendees: Optional[list[str]] = None,
        description: str = "",
    ) -> dict:
        """Create a calendar event. Write operation: gated like send_email."""
        self.require_enabled()
        event = {
            "summary": summary,
            "description": description,
            "start": {"dateTime": start_iso},
            "end": {"dateTime": end_iso},
        }
        if attendees:
            event["attendees"] = [{"email": a} for a in attendees]
        response = self._client.post(
            self.api,
            headers=self._headers(),
            json=event,
        )
        response.raise_for_status()
        return response.json()

    def status(self) -> dict:
        base = super().status()
        base["note"] = "Token needs calendar.events scope." if self.configured() else ""
        return base


def _scope_hint() -> str:
    return "gmail.send calendar.events youtube.upload"
