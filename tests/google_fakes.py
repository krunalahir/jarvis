"""Fakes for Google sign-in.

Nothing here touches the network or needs the real Google libraries.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import urlencode

GOOD_CODE = "good-code"  # the one-time code the fake Google accepts


class FakeGoogleSignIn:
    """Stands in for GoogleOAuth where a test only needs "signed in or not".

    token=None means nobody is signed in. granted=None means every
    permission was granted.
    """

    def __init__(self, token="tok", granted=None):
        self.token = token
        self.granted = granted

    def connected(self) -> bool:
        return self.token is not None

    def missing_scopes(self, scopes=()) -> list:
        if self.token is None:
            return list(scopes)
        if self.granted is None:
            return []
        return [scope for scope in scopes if scope not in self.granted]

    def access_token(self) -> str:
        if self.token is None:
            raise RuntimeError("Google is not connected.")
        return self.token


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def make_id_token(email: str) -> str:
    """A JWT-shaped string that carries an email, like Google's ID token."""

    def part(data: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")

    return f"{part({'alg': 'none'})}.{part({'email': email})}.signature"


class FakeCredentials:
    """The parts of google.oauth2.credentials.Credentials that JARVIS uses."""

    refresh_calls = 0

    def __init__(
        self,
        token,
        refresh_token="refresh-1",
        scopes=None,
        expiry=None,
        id_token=None,
        granted_scopes=None,
    ):
        self.token = token
        self.refresh_token = refresh_token
        self.scopes = scopes
        self.expiry = expiry or (_now() + timedelta(hours=1))
        self.id_token = id_token
        self.granted_scopes = granted_scopes

    @property
    def expired(self) -> bool:
        return self.expiry <= _now()

    @property
    def valid(self) -> bool:
        return bool(self.token) and not self.expired

    def to_json(self) -> str:
        return json.dumps(
            {
                "token": self.token,
                "refresh_token": self.refresh_token,
                "token_uri": "https://oauth2.googleapis.com/token",
                "client_id": "client-id",
                "client_secret": "client-secret",
                "scopes": self.scopes,
                "expiry": self.expiry.isoformat() + "Z",
            }
        )

    @classmethod
    def from_authorized_user_info(cls, info, scopes=None):
        missing = {"refresh_token", "client_id", "client_secret"} - set(info)
        if missing:
            raise ValueError(f"missing fields {sorted(missing)}")
        expiry = info.get("expiry")
        return cls(
            token=info.get("token"),
            refresh_token=info.get("refresh_token"),
            scopes=scopes or info.get("scopes"),
            expiry=datetime.fromisoformat(expiry.rstrip("Z")) if expiry else None,
        )

    def refresh(self, request) -> None:
        type(self).refresh_calls += 1
        self.token = f"renewed-{type(self).refresh_calls}"
        self.expiry = _now() + timedelta(hours=1)


class FakeFlow:
    """The parts of google_auth_oauthlib.flow.Flow that JARVIS uses.

    Like newer versions of the real library, it makes up a PKCE code
    verifier in authorization_url(). fetch_token() refuses to continue
    unless the same verifier is handed back, which is how the tests prove
    that JARVIS carries it from start() to finish().
    """

    granted = None  # scopes the fake Google grants; None = all requested
    email = "user@example.com"
    exchanges: list = []  # one entry per successful code exchange

    def __init__(self, client_config, scopes, state=None):
        self.client_config = client_config
        self.scopes = list(scopes)
        self.state = state
        self.redirect_uri = None
        self.code_verifier = None
        self.credentials = None

    @classmethod
    def reset(cls) -> None:
        cls.granted = None
        cls.email = "user@example.com"
        cls.exchanges = []
        FakeCredentials.refresh_calls = 0

    @classmethod
    def from_client_config(cls, client_config, scopes, **kwargs):
        return cls(client_config, scopes, state=kwargs.get("state"))

    def authorization_url(self, **kwargs):
        self.code_verifier = f"verifier-for-{self.state}"
        query = urlencode(
            {
                "client_id": self.client_config["web"]["client_id"],
                "redirect_uri": self.redirect_uri,
                "scope": " ".join(self.scopes),
                "state": self.state,
                **kwargs,
            }
        )
        return f"https://accounts.google.com/o/oauth2/auth?{query}", self.state

    def fetch_token(self, code=None, **kwargs):
        if code != GOOD_CODE:
            raise RuntimeError("(invalid_grant) Bad Request")
        if self.code_verifier != f"verifier-for-{self.state}":
            raise RuntimeError("(invalid_grant) Missing code verifier.")
        cls = type(self)
        cls.exchanges.append(
            {
                "code": code,
                "redirect_uri": self.redirect_uri,
                "code_verifier": self.code_verifier,
            }
        )
        self.credentials = FakeCredentials(
            token="access-1",
            scopes=list(self.scopes),
            id_token=make_id_token(cls.email),
            granted_scopes=list(self.scopes if cls.granted is None else cls.granted),
        )
