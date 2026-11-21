"""Google connectors: Gmail and Calendar.

They act as the Google account the user signed in with. The access token
comes from that sign-in (jarvis/google_oauth.py) and renews itself; there
is no GOOGLE_ACCESS_TOKEN to configure. YouTube (social.py) shares the
same sign-in through GoogleConnector.
"""

from __future__ import annotations

import base64
from email.message import EmailMessage
from typing import Optional

import httpx

from ..config import Settings
from ..google_oauth import CALENDAR_EVENTS, GMAIL_READONLY, GMAIL_SEND
from .base import Connector, ConnectorError, ConnectorNotConfigured


class GoogleConnector(Connector):
    """Base for connectors that use the user's Google sign-in.

    For these, "configured" means "the user has signed in to Google".
    `google_oauth` is the app's GoogleOAuth (or anything with the same
    connected() / missing_scopes() / access_token() methods).
    """

    # Every Google permission this connector may use (for the status note).
    scopes: tuple[str, ...] = ()

    def __init__(
        self,
        settings: Settings,
        client: Optional[httpx.Client] = None,
        google_oauth=None,
    ):
        super().__init__(settings, client)
        self.google_oauth = google_oauth

    def configured(self) -> bool:
        return self.google_oauth is not None and self.google_oauth.connected()

    def missing_settings(self) -> list[str]:
        # What sign-in itself still needs from .env. Empty once that is in
        # place, even before anyone has signed in (the status note says so).
        if self.configured():
            return []
        return self.settings.missing("google_client_id", "google_client_secret")

    def require_configured(self) -> None:
        if not self.configured():
            raise ConnectorNotConfigured(
                f"{self.name}: Google is not connected. The user must sign in "
                "first: use the connect_google tool (or ask JARVIS in chat to "
                "connect Google) and open the link it returns."
            )

    def status(self) -> dict:
        base = super().status()
        if not self.configured():
            base["note"] = (
                "Google is not connected. Sign in from chat: ask JARVIS to "
                "connect Google (the connect_google tool)."
            )
            return base
        missing = self.google_oauth.missing_scopes(self.scopes)
        if missing:
            base["note"] = (
                "Signed in to Google, but these permissions were not granted: "
                + ", ".join(missing)
            )
        else:
            base["note"] = "Signed in to Google."
        return base

    def _headers(self, *scopes: str) -> dict:
        """Authorization header for the signed-in account.

        `scopes` are the Google permissions this call needs. If the user
        left one unticked on Google's consent screen, say so here instead
        of letting Google answer with a bare 403.
        """
        self.require_configured()
        missing = self.google_oauth.missing_scopes(scopes)
        if missing:
            raise ConnectorNotConfigured(
                f"{self.name}: Google is connected, but this permission was "
                f"not granted: {', '.join(missing)}. Connect Google again "
                "(connect_google with reconnect=true) and tick every box."
            )
        try:
            token = self.google_oauth.access_token()
        except Exception as exc:  # expired, revoked, damaged token file...
            raise ConnectorNotConfigured(f"{self.name}: {exc}") from exc
        return {"Authorization": f"Bearer {token}"}

    def _check(self, response: httpx.Response) -> None:
        """Raise a ConnectorError that carries Google's own explanation.

        raise_for_status() would only say "403 Forbidden"; Google's reply
        says why (for example that the Gmail API is not enabled yet).
        """
        if response.status_code < 400:
            return
        try:
            message = response.json()["error"]["message"]
        except Exception:
            message = response.text[:200]
        raise ConnectorError(
            f"{self.name}: Google API error {response.status_code}: {message}"
        )


class GmailConnector(GoogleConnector):
    name = "gmail"
    scopes = (GMAIL_READONLY, GMAIL_SEND)
    api = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"

    def search_messages(self, query: str, limit: int = 10) -> list[dict]:
        """Read operation: needs the Google sign-in, not the actions switch."""
        response = self._client.get(
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            headers=self._headers(GMAIL_READONLY),
            params={"q": query, "maxResults": limit},
        )
        self._check(response)
        return response.json().get("messages", [])

    def send_email(self, to: str, subject: str, body: str, cc: Optional[list[str]] = None) -> dict:
        """Send an email as the signed-in Google account. This is a write:
        it requires the sign-in and JARVIS_ACTIONS_ENABLED=true."""
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
            headers=self._headers(GMAIL_SEND),
            json={"raw": raw},
        )
        self._check(response)
        return response.json()


class CalendarConnector(GoogleConnector):
    name = "google_calendar"
    scopes = (CALENDAR_EVENTS,)
    api = "https://www.googleapis.com/calendar/v3/calendars/primary/events"

    def list_events(self, time_min_iso: Optional[str] = None, limit: int = 10) -> list[dict]:
        """Read operation: needs the Google sign-in, not the actions switch."""
        params: dict = {"maxResults": limit, "singleEvents": "true", "orderBy": "startTime"}
        if time_min_iso:
            params["timeMin"] = time_min_iso
        response = self._client.get(
            self.api, headers=self._headers(CALENDAR_EVENTS), params=params
        )
        self._check(response)
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
            headers=self._headers(CALENDAR_EVENTS),
            json=event,
        )
        self._check(response)
        return response.json()
