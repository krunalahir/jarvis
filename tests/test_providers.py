"""Provider response-parsing tests using mock HTTP transports - no network."""

import httpx

from jarvis.research.providers import (
    ExaProvider,
    OpenAIWebSearchProvider,
    PerplexityProvider,
)


def mock_client(payload: dict, status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_provider_extracts_text_and_citations():
    payload = {
        "output": [
            {
                "type": "message",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Ahmedabad is in Gujarat. [1]",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Ahmedabad",
                                "url": "https://example.com/amd",
                            }
                        ],
                    }
                ],
            }
        ]
    }
    provider = OpenAIWebSearchProvider("key", client=mock_client(payload))
    result = provider.research("Where is Ahmedabad?")
    assert result.sources[0].url == "https://example.com/amd"
    assert result.claims[0].source_indexes == [0]


def test_perplexity_provider_maps_citation_list():
    payload = {
        "choices": [{"message": {"content": "It rains in July. [1] Usually. [2]"}}],
        "citations": ["https://a.example", "https://b.example"],
    }
    provider = PerplexityProvider("key", client=mock_client(payload))
    result = provider.research("weather?")
    assert len(result.sources) == 2
    assert result.claims[0].source_indexes == [0]
    assert result.claims[1].source_indexes == [1]


def test_exa_provider_returns_sources_only():
    payload = {
        "results": [
            {"title": "T", "url": "https://a.example", "publishedDate": "2026-01-01", "text": "body"}
        ]
    }
    provider = ExaProvider("key", client=mock_client(payload))
    result = provider.research("anything")
    assert result.summary == ""
    assert result.sources[0].title == "T"
    assert result.claims[0].source_indexes == [0]


def test_provider_error_on_http_failure():
    from jarvis.research.providers import ProviderError

    provider = OpenAIWebSearchProvider("key", client=mock_client({"error": "bad"}, status=401))
    try:
        provider.research("q")
        assert False, "expected ProviderError"
    except ProviderError as exc:
        assert "401" in str(exc)


def test_background_job_completes():
    from jarvis.research import ResearchPipeline, StaticProvider, Store
    from jarvis.research.jobs import JobRunner
    from jarvis.research.providers import ProviderResult
    from jarvis.schemas import Claim, Source

    store = Store(":memory:")
    pipeline = ResearchPipeline(
        StaticProvider(
            default=ProviderResult(
                summary="s [1]",
                claims=[Claim(text="s", source_indexes=[0])],
                sources=[Source(title="t", url="https://example.com")],
            )
        ),
        store,
    )
    runner = JobRunner(pipeline, store)
    job = runner.submit("test question")
    # wait for the worker thread
    runner._pool.shutdown(wait=True)
    done = runner.get(job.id)
    assert done.status.value == "done"
    assert store.get_answer(done.answer_id).question == "test question"


def test_scheduler_publish_success_path():
    from jarvis.research import Store
    from jarvis.scheduler import PublishScheduler
    from jarvis.schemas import DraftStatus, Platform

    class FakeConnector:
        name = "x"

        def publish_draft(self, draft):
            return {"id": "posted-1"}

    store = Store(":memory:")
    scheduler = PublishScheduler(store, {"x": FakeConnector()})
    from datetime import datetime, timedelta, timezone

    draft = scheduler.create_draft(Platform.X, "hello")
    scheduler.schedule(draft.id, datetime.now(timezone.utc) - timedelta(seconds=1))
    touched = scheduler.publish_due()
    assert touched[0].status == DraftStatus.PUBLISHED
    assert touched[0].platform_post_id == "posted-1"
