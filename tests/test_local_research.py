"""Local Qwen3 research provider tests - mock HTTP for search, scripted
model for synthesis. No network, no real model."""

import httpx
import pytest
from langchain_core.messages import AIMessage

from jarvis.config import Settings
from jarvis.research import ResearchPipeline, Store, provider_from_settings
from jarvis.research.providers import (
    ExaProvider,
    LocalQwenResearchProvider,
    PerplexityProvider,
    ProviderError,
)
from jarvis.schemas import AnswerStatus

from fakes import ScriptedChatModel

EXA_PAYLOAD = {
    "results": [
        {
            "title": "Ahmedabad",
            "url": "https://example.com/amd",
            "publishedDate": "2026-11-01",
            "text": "Ahmedabad is the largest city in Gujarat.",
        }
    ]
}


def exa_client(payload=EXA_PAYLOAD, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_local_provider_synthesizes_only_from_search_sources():
    settings = Settings(exa_api_key="key")
    model = ScriptedChatModel(script=[AIMessage(content="Ahmedabad is in Gujarat. [1]")])
    provider = LocalQwenResearchProvider(settings, model=model, client=exa_client())
    result = provider.research("Where is Ahmedabad?")
    assert result.sources[0].url == "https://example.com/amd"
    assert result.claims[0].source_indexes == [0]


def test_local_provider_needs_search_key():
    settings = Settings()
    provider = LocalQwenResearchProvider(settings, model=ScriptedChatModel(script=[]))
    assert provider.available() is False
    store = Store(":memory:")
    answer = ResearchPipeline(provider, store).ask("anything fresh?")
    store.close()
    assert answer.status == AnswerStatus.NEEDS_CONFIGURATION
    assert "EXA_API_KEY" in answer.detail
    assert "cannot browse the web" in answer.detail


def test_no_search_results_stays_uncertain_never_invented():
    settings = Settings(exa_api_key="key")
    model = ScriptedChatModel(script=[AIMessage(content="should not be used")])
    provider = LocalQwenResearchProvider(settings, model=model, client=exa_client({"results": []}))
    store = Store(":memory:")
    answer = ResearchPipeline(provider, store).ask("obscure question")
    store.close()
    assert answer.status == AnswerStatus.UNCERTAIN
    assert model.calls == 0  # the model is never asked without sources


def test_unreachable_model_server_is_a_clear_error():
    settings = Settings(exa_api_key="key")
    model = ScriptedChatModel(script=[ConnectionError("refused")])
    provider = LocalQwenResearchProvider(settings, model=model, client=exa_client())
    with pytest.raises(ProviderError) as exc:
        provider.research("anything")
    assert "local model server" in str(exc.value)
    assert "docs/local-model.md" in str(exc.value)


def test_default_provider_is_local_and_hosted_is_opt_in():
    default = provider_from_settings(Settings())
    assert isinstance(default, LocalQwenResearchProvider)
    assert isinstance(
        provider_from_settings(Settings(research_provider="perplexity", perplexity_api_key="k")),
        PerplexityProvider,
    )
    assert isinstance(
        provider_from_settings(Settings(research_provider="exa", exa_api_key="k")),
        ExaProvider,
    )
