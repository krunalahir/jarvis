"""Connector registry: one place to see what's connected and ready."""

from .base import (
    ActionsDisabled,
    Connector,
    ConnectorError,
    ConnectorNotConfigured,
    PlatformAccessRequired,
)
from .calling import TwilioCallingConnector
from .google_workspace import CalendarConnector, GmailConnector, GoogleConnector
from .social import (
    InstagramConnector,
    SnapchatConnector,
    XConnector,
    YouTubeConnector,
)
from .spotify import SpotifyConnector

ALL_CONNECTORS = [
    GmailConnector,
    CalendarConnector,
    TwilioCallingConnector,
    InstagramConnector,
    XConnector,
    YouTubeConnector,
    SnapchatConnector,
    SpotifyConnector,
]


def build_connectors(
    settings, clients: dict | None = None, google_oauth=None
) -> dict[str, Connector]:
    """Instantiate every connector. `clients` lets tests inject mock HTTP.
    `google_oauth` is the one Google sign-in shared by Gmail, Calendar and
    YouTube."""
    clients = clients or {}
    connectors = {}
    for cls in ALL_CONNECTORS:
        extra = {"google_oauth": google_oauth} if issubclass(cls, GoogleConnector) else {}
        connectors[cls.name] = cls(settings, client=clients.get(cls.name), **extra)
    return connectors


__all__ = [
    "ActionsDisabled",
    "Connector",
    "ConnectorError",
    "ConnectorNotConfigured",
    "PlatformAccessRequired",
    "GoogleConnector",
    "GmailConnector",
    "CalendarConnector",
    "TwilioCallingConnector",
    "InstagramConnector",
    "XConnector",
    "YouTubeConnector",
    "SnapchatConnector",
    "SpotifyConnector",
    "ALL_CONNECTORS",
    "build_connectors",
]
