"""Connector base classes.

Every connector reports its own status and refuses to act unless it is
both configured (platform credentials present) and enabled (the global
JARVIS_ACTIONS_ENABLED switch is on). Reads like status checks only need
configuration; writes need both.
"""

from __future__ import annotations

from typing import Optional

import httpx

from ..config import Settings


class ConnectorError(Exception):
    pass


class ConnectorNotConfigured(ConnectorError):
    pass


class ActionsDisabled(ConnectorError):
    pass


class PlatformAccessRequired(ConnectorError):
    """The platform does not offer this capability through a public API."""


class Connector:
    name = "connector"
    required_settings: tuple[str, ...] = ()

    def __init__(self, settings: Settings, client: Optional[httpx.Client] = None):
        self.settings = settings
        self._client = client or httpx.Client(timeout=30)

    def configured(self) -> bool:
        return self.settings.has_all(*self.required_settings)

    def missing_settings(self) -> list[str]:
        return self.settings.missing(*self.required_settings)

    def enabled(self) -> bool:
        return self.configured() and self.settings.actions_enabled

    def status(self) -> dict:
        if not self.configured():
            state = "not_configured"
        elif not self.settings.actions_enabled:
            state = "configured_actions_disabled"
        else:
            state = "ready"
        return {
            "name": self.name,
            "state": state,
            "missing_settings": self.missing_settings(),
            "actions_enabled": self.settings.actions_enabled,
        }

    def require_configured(self) -> None:
        if not self.configured():
            raise ConnectorNotConfigured(
                f"{self.name}: missing settings: {', '.join(self.missing_settings())}"
            )

    def require_enabled(self) -> None:
        self.require_configured()
        if not self.settings.actions_enabled:
            raise ActionsDisabled(
                f"{self.name}: JARVIS_ACTIONS_ENABLED is false; refusing to act."
            )
