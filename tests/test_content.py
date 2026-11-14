"""Content generation tests with a mock transport - no network."""

import httpx
import pytest

from jarvis.config import Settings
from jarvis.content import ContentGenerationUnavailable, ContentGenerator
from jarvis.schemas import Platform


def mock_client(payload: dict, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_generate_returns_model_text():
    payload = {"choices": [{"message": {"content": "Launch day is here!"}}]}
    generator = ContentGenerator(Settings(openai_api_key="key"), client=mock_client(payload))
    text = generator.generate(Platform.X, "product launch")
    assert text == "Launch day is here!"


def test_generate_unavailable_without_key():
    generator = ContentGenerator(Settings())
    with pytest.raises(ContentGenerationUnavailable):
        generator.generate(Platform.X, "anything")
