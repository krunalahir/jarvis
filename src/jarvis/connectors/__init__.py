"""Connector registry: one place to see what's connected and ready."""

from .base import (
    ActionsDisabled,
    Connector,
    ConnectorError,
    ConnectorNotConfigured,
    PlatformAccessRequired,
)
from .calling import TwilioCallingConnector
from .google_workspace import CalendarConnector, GmailConnector
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


def build_connectors(settings, clients: dict | None = None) -> dict[str, Connector]:
    """Instantiate every connector. `clients` lets tests inject mock HTTP."""
    clients = clients or {}
    return {
        cls.name: cls(settings, client=clients.get(cls.name))
        for cls in ALL_CONNECTORS
    }


__all__ = [
    "ActionsDisabled",
    "Connector",
    "ConnectorError",
    "ConnectorNotConfigured",
    "PlatformAccessRequired",
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
