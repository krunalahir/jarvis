"""LLM configuration tests."""

import pytest
from langchain_openai import ChatOpenAI

from jarvis.config import Settings
from jarvis.llm import LLMNotConfigured, build_chat_model, describe_llm


def test_default_is_local_qwen():
    settings = Settings()
    assert settings.llm_provider == "local"
    assert settings.local_llm_model == "qwen3-8b"
    model = build_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.openai_api_base == settings.local_llm_base_url


def test_describe_local_llm():
    info = describe_llm(Settings())
    assert info["provider"] == "local"
    assert "qwen3-8b" in info["model"]


def test_openai_opt_in_needs_key():
    with pytest.raises(LLMNotConfigured):
        build_chat_model(Settings(llm_provider="openai"))


def test_unknown_provider_rejected():
    with pytest.raises(LLMNotConfigured):
        build_chat_model(Settings(llm_provider="anthropic"))
