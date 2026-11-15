"""Content generation tests with a scripted model - no network."""

import pytest
from langchain_core.messages import AIMessage

from jarvis.config import Settings
from jarvis.content import ContentGenerationUnavailable, ContentGenerator
from jarvis.schemas import Platform

from fakes import ScriptedChatModel


def test_generate_returns_model_text():
    model = ScriptedChatModel(script=[AIMessage(content="Launch day is here!")])
    generator = ContentGenerator(Settings(), model=model)
    text = generator.generate(Platform.X, "product launch")
    assert text == "Launch day is here!"


def test_local_is_available_without_any_key():
    generator = ContentGenerator(Settings())
    assert generator.available() is True


def test_openai_opt_in_without_key_is_unavailable():
    generator = ContentGenerator(Settings(llm_provider="openai"))
    assert generator.available() is False
    with pytest.raises(ContentGenerationUnavailable) as exc:
        generator.generate(Platform.X, "anything")
    assert "OPENAI_API_KEY" in str(exc.value)


def test_unreachable_local_server_is_a_clear_error():
    model = ScriptedChatModel(script=[ConnectionError("refused")])
    generator = ContentGenerator(Settings(), model=model)
    with pytest.raises(ContentGenerationUnavailable) as exc:
        generator.generate(Platform.X, "anything")
    assert "local model server" in str(exc.value)
    assert "docs/local-model.md" in str(exc.value)
