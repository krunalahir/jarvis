"""Chat model construction (LangChain).

JARVIS runs on a self-hosted Qwen3 8B by default. Any serving tool that
exposes an OpenAI-compatible chat endpoint works: LM Studio, Ollama,
mlx-lm or vLLM. LangChain's ChatOpenAI speaks that protocol, so the same
code drives the local model and (with explicit opt-in) hosted OpenAI.

Setup for the local server: docs/local-model.md.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from .config import Settings

OPENAI_MODEL = "gpt-4.1-mini"


class LLMNotConfigured(Exception):
    pass


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Return the LangChain chat model for the configured provider.

    "local" (default) points at the user's own Qwen3 8B server; it needs
    no API key and no account. "openai" is a paid, hosted opt-in and needs
    OPENAI_API_KEY.
    """
    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise LLMNotConfigured(
                "JARVIS_LLM_PROVIDER=openai needs OPENAI_API_KEY. Leave the "
                "provider as 'local' to use your self-hosted Qwen3 8B instead."
            )
        return ChatOpenAI(model=OPENAI_MODEL, api_key=settings.openai_api_key)
    if settings.llm_provider != "local":
        raise LLMNotConfigured(
            f"Unknown JARVIS_LLM_PROVIDER: {settings.llm_provider!r} "
            "(expected 'local' or 'openai')."
        )
    return ChatOpenAI(
        model=settings.local_llm_model,
        base_url=settings.local_llm_base_url,
        api_key=settings.local_llm_api_key or "EMPTY",
        temperature=0.2,
    )


def describe_llm(settings: Settings) -> dict:
    """Status payload: which model JARVIS will use and where."""
    if settings.llm_provider == "openai":
        return {
            "provider": "openai",
            "model": OPENAI_MODEL,
            "endpoint": "https://api.openai.com/v1",
            "note": "Hosted and paid. Opted in via JARVIS_LLM_PROVIDER=openai.",
        }
    return {
        "provider": "local",
        "model": settings.local_llm_model,
        "endpoint": settings.local_llm_base_url,
        "note": "Self-hosted Qwen3 8B over an OpenAI-compatible local server. "
                "See docs/local-model.md.",
    }
