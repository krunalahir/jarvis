import pytest

from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult, UnconfiguredProvider
from jarvis.schemas import AnswerStatus, Claim, Source


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


SOURCED = ProviderResult(
    summary="Ahmedabad is the largest city in Gujarat. [1]",
    claims=[Claim(text="Ahmedabad is the largest city in Gujarat.", source_indexes=[0])],
    sources=[Source(title="Ahmedabad", url="https://example.com/ahmedabad")],
)


def test_answer_with_sources_is_answered_and_saved(store):
    pipeline = ResearchPipeline(StaticProvider(default=SOURCED), store)
    answer = pipeline.ask("What is the largest city in Gujarat?")
    assert answer.status == AnswerStatus.ANSWERED
    assert answer.claims and answer.sources
    assert store.get_answer(answer.id).question == answer.question


def test_no_sources_marks_uncertain(store):
    result = ProviderResult(summary="Maybe 42.", claims=[Claim(text="Maybe 42.")], sources=[])
    pipeline = ResearchPipeline(StaticProvider(default=result), store)
    answer = pipeline.ask("anything")
    assert answer.status == AnswerStatus.UNCERTAIN
    assert "no sources" in answer.detail


def test_unsupported_claims_are_surfaced_not_dropped(store):
    result = ProviderResult(
        summary="A [1]. B.",
        claims=[
            Claim(text="A", source_indexes=[0]),
            Claim(text="B", source_indexes=[]),
        ],
        sources=[Source(title="t", url="https://example.com")],
    )
    pipeline = ResearchPipeline(StaticProvider(default=result), store)
    answer = pipeline.ask("anything")
    assert answer.status == AnswerStatus.UNCERTAIN
    assert [c.text for c in answer.claims] == ["A"]
    assert answer.unsupported_claims == ["B"]


def test_unconfigured_provider_is_explicit(store):
    pipeline = ResearchPipeline(UnconfiguredProvider(), store)
    answer = pipeline.ask("anything")
    assert answer.status == AnswerStatus.NEEDS_CONFIGURATION
    assert "OPENAI_API_KEY" in answer.detail


def test_empty_question_rejected(store):
    pipeline = ResearchPipeline(StaticProvider(default=SOURCED), store)
    with pytest.raises(ValueError):
        pipeline.ask("   ")


def test_citation_extraction():
    from jarvis.research.providers import claims_from_cited_text

    claims = claims_from_cited_text("Sky is blue. [1] Water is wet. [9]", source_count=1)
    assert claims[0].source_indexes == [0]
    assert claims[1].source_indexes == []  # [9] out of range -> unsupported
