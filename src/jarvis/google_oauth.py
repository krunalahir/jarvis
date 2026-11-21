"""Google sign-in for local, single-user JARVIS.

How the pieces fit together:

1. In chat, the user asks JARVIS to connect Google. The agent's
   connect_google tool calls start() and hands back a sign-in link.
2. The user opens the link and presses Allow. Only the user can do this.
3. Google sends the browser to GOOGLE_REDIRECT_URI, which is this app's
   /auth/google/callback route. finish() swaps the one-time code for
   tokens and saves them in GOOGLE_TOKEN_FILE.
4. From then on Gmail, Calendar and YouTube call access_token(). The
   token renews itself, so there is no GOOGLE_ACCESS_TOKEN to paste.

The token file holds a refresh token and the client secret. Treat it like
a password and keep it out of git.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

try:  # Optional at import time, so the rest of JARVIS runs without them.
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
except ImportError:  # reported by GoogleOAuth.problem(), not at import
    Request = Credentials = Flow = None  # type: ignore[assignment,misc]

GMAIL_READONLY = "https://www.googleapis.com/auth/gmail.readonly"
GMAIL_SEND = "https://www.googleapis.com/auth/gmail.send"
CALENDAR_EVENTS = "https://www.googleapis.com/auth/calendar.events"
YOUTUBE_UPLOAD = "https://www.googleapis.com/auth/youtube.upload"

# Asked for only to learn which account signed in. Full scope URLs on
# purpose: Google answers the short alias "email" with the long form, and
# the OAuth library treats that difference as an error.
IDENTITY_SCOPES = ["openid", "https://www.googleapis.com/auth/userinfo.email"]
# What JARVIS needs to do its work. Only these are ever reported missing.
FEATURE_SCOPES = [GMAIL_READONLY, GMAIL_SEND, CALENDAR_EVENTS, YOUTUBE_UPLOAD]
SCOPES = IDENTITY_SCOPES + FEATURE_SCOPES

# How long a sign-in link stays usable.
SIGN_IN_LINK_TTL_SECONDS = 600
# Unfinished sign-ins remembered at once (oldest dropped first).
_MAX_PENDING_SIGN_INS = 20


class GoogleSignInError(Exception):
    """Sign-in cannot continue. The message says what the user should do."""


class GoogleOAuth:
    def __init__(self, settings):
        self.client_id = settings.google_client_id
        self.client_secret = settings.google_client_secret
        self.redirect_uri = settings.google_redirect_uri
        self.token_file = Path(settings.google_token_file)
        self._lock = threading.RLock()
        # state -> (expires_at, PKCE code verifier or None). Kept in memory,
        # which is right for one local process. Use shared storage instead
        # if you ever run several workers or machines.
        self._pending: dict[str, tuple[float, Optional[str]]] = {}

    # --- setup ---

    def problem(self) -> Optional[str]:
        """Why sign-in cannot start right now, or None when it can."""
        if Flow is None or Credentials is None:
            return (
                "the google-auth-oauthlib package is not installed. Run "
                "'pip install -e .' in the project folder."
            )
        missing = [
            name
            for name, value in (
                ("GOOGLE_CLIENT_ID", self.client_id),
                ("GOOGLE_CLIENT_SECRET", self.client_secret),
            )
            if not value
        ]
        if missing:
            return f"{' and '.join(missing)} must be set in .env."
        return None

    def _client_config(self) -> dict:
        return {
            "web": {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    # --- the sign-in itself ---

    def start(self) -> str:
        """Begin a sign-in and return the link the user must open."""
        problem = self.problem()
        if problem:
            raise GoogleSignInError(f"Google sign-in is not set up yet: {problem}")
        state = secrets.token_urlsafe(32)
        flow = Flow.from_client_config(self._client_config(), scopes=SCOPES, state=state)
        flow.redirect_uri = self.redirect_uri
        link, _ = flow.authorization_url(
            access_type="offline",  # ask for a refresh token
            include_granted_scopes="true",
            prompt="consent",  # so Google returns the refresh token every time
        )
        with self._lock:
            now = time.time()
            self._pending = {s: p for s, p in self._pending.items() if p[0] > now}
            while len(self._pending) >= _MAX_PENDING_SIGN_INS:
                self._pending.pop(next(iter(self._pending)))
            # Some versions of the Google library create a one-time PKCE
            # code verifier here. finish() runs on a new Flow object, so
            # keep the verifier or Google rejects the code exchange.
            self._pending[state] = (
                now + SIGN_IN_LINK_TTL_SECONDS,
                getattr(flow, "code_verifier", None),
            )
        return link

    def finish(self, code: str, state: str) -> dict:
        """Complete the sign-in Google redirected back with. Returns status()."""
        with self._lock:
            expires_at, code_verifier = self._pending.pop(state, (0.0, None))
        if expires_at < time.time():
            raise GoogleSignInError(
                "This sign-in link is unknown or has expired. Ask JARVIS to "
                "connect Google again and use the new link."
            )
        # Google may grant more scopes than asked (ones approved earlier) or
        # fewer (the user can untick boxes). The OAuth library raises on any
        # difference unless this is set; missing ones are reported by
        # missing_scopes() instead.
        os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
        flow = Flow.from_client_config(self._client_config(), scopes=SCOPES, state=state)
        flow.redirect_uri = self.redirect_uri
        if code_verifier:
            flow.code_verifier = code_verifier
        flow.fetch_token(code=code)
        creds = flow.credentials
        info = json.loads(creds.to_json())
        # Keep what Google actually granted: it is what a token refresh may
        # ask for, and what missing_scopes() checks against.
        info["scopes"] = _granted_scopes(flow, creds)
        info["email"] = _email_from_id_token(getattr(creds, "id_token", None))
        with self._lock:
            self._write(info)
        return self.status()

    # --- after sign-in ---

    def connected(self) -> bool:
        """True once the user has signed in. Reads the token file; no network."""
        return self._read() is not None

    def missing_scopes(self, scopes: Iterable[str] = FEATURE_SCOPES) -> list[str]:
        """Which of `scopes` the user did not grant (all, if not signed in)."""
        info = self._read()
        if info is None:
            return list(scopes)
        granted = info.get("scopes")
        if not granted:  # unknown, e.g. an older token file: let Google decide
            return []
        return [scope for scope in scopes if scope not in granted]

    def access_token(self) -> str:
        """A valid access token for the signed-in account, renewed if needed."""
        with self._lock:
            info = self._read()
            if info is None:
                raise GoogleSignInError(
                    "Google is not connected. Ask JARVIS to connect Google first."
                )
            if Credentials is None:
                raise GoogleSignInError(f"Google sign-in is not usable: {self.problem()}")
            try:
                creds = Credentials.from_authorized_user_info(info)
            except ValueError as exc:
                raise GoogleSignInError(
                    "The saved Google sign-in is damaged. Ask JARVIS to "
                    "connect Google again."
                ) from exc
            if not creds.valid:
                if not creds.refresh_token:
                    raise GoogleSignInError(
                        "The Google sign-in has expired. Ask JARVIS to connect "
                        "Google again."
                    )
                try:
                    creds.refresh(Request())
                except Exception as exc:  # revoked, expired, network down...
                    raise GoogleSignInError(
                        f"Google would not renew the sign-in ({exc}). Ask "
                        "JARVIS to connect Google again."
                    ) from exc
                renewed = json.loads(creds.to_json())
                renewed["scopes"] = info.get("scopes")
                renewed["email"] = info.get("email")
                self._write(renewed)
            return creds.token

    def status(self) -> dict:
        """Connection summary. Reads the token file; no network call."""
        info = self._read()
        return {
            "connected": info is not None,
            "email": info.get("email") if info else None,
            "missing_scopes": self.missing_scopes(),
            "problem": self.problem(),
        }

    # --- token file ---

    def _read(self) -> Optional[dict]:
        try:
            info = json.loads(self.token_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(info, dict):
            return None
        if not (info.get("refresh_token") or info.get("token")):
            return None
        return info

    def _write(self, info: dict) -> None:
        """Write the token file so only this user can read it, and so a
        reader never sees a half-written file."""
        self.token_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.token_file.with_name(self.token_file.name + ".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(info, handle)
        try:
            os.chmod(tmp, 0o600)  # no effect on Windows
        except OSError:
            pass
        os.replace(tmp, self.token_file)


def _granted_scopes(flow, creds) -> list[str]:
    """The scopes Google actually granted in this sign-in."""
    granted = getattr(creds, "granted_scopes", None)
    if not granted:
        token = getattr(getattr(flow, "oauth2session", None), "token", None) or {}
        granted = token.get("scope")
    if isinstance(granted, str):
        granted = granted.split()
    return sorted(set(granted or getattr(creds, "scopes", None) or SCOPES))


def _email_from_id_token(id_token: Optional[str]) -> Optional[str]:
    """Read the account's email address out of Google's ID token.

    The token came straight from Google's token endpoint over HTTPS, so
    its payload is read here without re-checking the signature. It is used
    only to show which account is connected.
    """
    if not id_token:
        return None
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        return json.loads(base64.urlsafe_b64decode(payload)).get("email")
    except (IndexError, ValueError, AttributeError):
        return None
