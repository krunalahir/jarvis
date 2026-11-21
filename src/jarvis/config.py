"""Configuration and credential gating for JARVIS.

Everything comes from the environment. No secrets are stored in the
repository; see .env.example for the full list.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@lru_cache(maxsize=1)
def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from .env without overriding real env vars."""
    root = Path(__file__).resolve().parents[2]
    for path in dict.fromkeys([Path.cwd() / ".env", root / ".env"]):
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ.setdefault(key, value)
        return


def _env(name: str) -> Optional[str]:
    value = os.environ.get(name, "").strip()
    return value or None


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    """Runtime settings. Read with Settings.from_env()."""

    actions_enabled: bool = False
    api_key: Optional[str] = None
    store_path: str = "jarvis_store.sqlite3"

    openai_api_key: Optional[str] = None
    perplexity_api_key: Optional[str] = None
    exa_api_key: Optional[str] = None

    # --- Model (LLM) settings ---
    # "local" (default): a self-hosted Qwen3 8B served over an
    # OpenAI-compatible endpoint (LM Studio, Ollama, MLX or vLLM).
    # "openai": hosted OpenAI, explicit opt-in only.
    llm_provider: str = "local"
    local_llm_base_url: str = "http://127.0.0.1:1234/v1"
    local_llm_model: str = "qwen3-8b"
    local_llm_api_key: str = "EMPTY"
    # Research provider: unset means "local" (Qwen3 synthesis + Exa web
    # search). Set JARVIS_RESEARCH_PROVIDER to openai, perplexity or exa
    # to opt into a hosted provider instead.
    research_provider: Optional[str] = None
    # Agent loop bound: maximum plan/tool rounds per chat request.
    agent_max_steps: int = 8

    google_access_token: Optional[str] = None

    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None

    meta_access_token: Optional[str] = None
    instagram_business_account_id: Optional[str] = None

    x_bearer_token: Optional[str] = None

    spotify_access_token: Optional[str] = None

    # Escape hatch for running with actions enabled and no API key
    # (local-only demos). Never set this when the API is reachable by
    # anyone else.
    allow_insecure: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            actions_enabled=_flag("JARVIS_ACTIONS_ENABLED"),
            api_key=_env("JARVIS_API_KEY"),
            store_path=os.environ.get("JARVIS_STORE_PATH", "jarvis_store.sqlite3"),
            openai_api_key=_env("OPENAI_API_KEY"),
            perplexity_api_key=_env("PERPLEXITY_API_KEY"),
            exa_api_key=_env("EXA_API_KEY"),
            llm_provider=os.environ.get("JARVIS_LLM_PROVIDER", "local").strip().lower() or "local",
            local_llm_base_url=os.environ.get(
                "JARVIS_LOCAL_LLM_BASE_URL", "http://127.0.0.1:1234/v1"
            ).strip(),
            local_llm_model=os.environ.get("JARVIS_LOCAL_LLM_MODEL", "qwen3-8b").strip(),
            local_llm_api_key=os.environ.get("JARVIS_LOCAL_LLM_API_KEY", "EMPTY").strip(),
            research_provider=_env("JARVIS_RESEARCH_PROVIDER"),
            agent_max_steps=int(os.environ.get("JARVIS_AGENT_MAX_STEPS", "8")),
            google_access_token=_env("GOOGLE_ACCESS_TOKEN"),
            twilio_account_sid=_env("TWILIO_ACCOUNT_SID"),
            twilio_auth_token=_env("TWILIO_AUTH_TOKEN"),
            twilio_from_number=_env("TWILIO_FROM_NUMBER"),
            meta_access_token=_env("META_ACCESS_TOKEN"),
            instagram_business_account_id=_env("INSTAGRAM_BUSINESS_ACCOUNT_ID"),
            x_bearer_token=_env("X_BEARER_TOKEN"),
            spotify_access_token=_env("SPOTIFY_ACCESS_TOKEN"),
            allow_insecure=_flag("JARVIS_ALLOW_INSECURE"),
        )

    def security_warnings(self) -> list[str]:
        warnings = []
        if self.actions_enabled and not self.api_key:
            warnings.append(
                "JARVIS_ACTIONS_ENABLED is true but JARVIS_API_KEY is unset: "
                "anyone who can reach the API could request actions. Set an "
                "API key, or bind to 127.0.0.1 and set JARVIS_ALLOW_INSECURE=true."
            )
        if not self.api_key:
            warnings.append("JARVIS_API_KEY is unset: the HTTP API is unauthenticated.")
        return warnings

    def has_all(self, *names: str) -> bool:
        return all(getattr(self, name) for name in names)

    def missing(self, *names: str) -> list[str]:
        return [name for name in names if not getattr(self, name)]
