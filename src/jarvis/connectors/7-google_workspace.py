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
        """Read operation: needs configuration, not the actions switch.

        Returns only message/thread id stubs - Gmail's list endpoint
        carries no content. Use read_messages() when the assistant needs
        to actually read the mail.
        """
        self.require_configured()
        response = self._client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            headers=self._headers(),
            params={"q": query, "maxResults": limit},
        )
        response.raise_for_status()
        return response.json().get("messages", [])

    def get_message(self, message_id: str) -> dict:
        """Fetch one message with sender, subject, date and body text."""
        self.require_configured()
        response = self._client.get(
            f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}",
            headers=self._headers(),
            params={"format": "full"},
        )
        response.raise_for_status()
        data = response.json()
        payload = data.get("payload", {})
        headers = {
            h.get("name", "").lower(): h.get("value", "")
            for h in payload.get("headers", [])
        }
        return {
            "id": data.get("id", message_id),
            "from": headers.get("from", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "snippet": data.get("snippet", ""),
            "body": _plain_text_body(payload),
        }

    def read_messages(self, query: str, limit: int = 5) -> list[dict]:
        """Search, then fetch each message so results carry real content."""
        stubs = self.search_messages(query, limit=limit)
        return [self.get_message(m["id"]) for m in stubs if m.get("id")]

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


def _plain_text_body(payload: dict) -> str:
    """Walk a Gmail message payload for its text/plain body."""
    mime = payload.get("mimeType", "")
    body = payload.get("body", {})
    if mime == "text/plain" and body.get("data"):
        return base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="replace")
    for part in payload.get("parts", []):
        text = _plain_text_body(part)
        if text:
            return text
    if body.get("data") and not payload.get("parts"):
        return base64.urlsafe_b64decode(body["data"]).decode("utf-8", errors="replace")
    return ""


def _scope_hint() -> str:
    return "gmail.send calendar.events youtube.upload"

