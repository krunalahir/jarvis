"""Approval gate, durable jobs, citation validation, duplicate prevention."""

import os

import pytest

from jarvis.actions import ActionGate
from jarvis.config import Settings
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.jobs import JobRunner
from jarvis.research.providers import ProviderResult
from jarvis.schemas import (
    ActionKind,
    AnswerStatus,
    Claim,
    JobStatus,
    Platform,
    Source,
)


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


# --- approval gate ---


def test_gate_full_cycle(store):
    calls = []
    gate = ActionGate(store, {ActionKind.SPOTIFY_PLAY: lambda p: calls.append(p) or {"ok": 1}})
    action = gate.request(ActionKind.SPOTIFY_PLAY, "Play song", {"track_query": "song"})
    assert action.status.value == "pending"
    assert calls == []  # nothing executed at request time

    executed = gate.approve(action.id)
    assert executed.status.value == "executed"
    assert calls == [{"track_query": "song"}]

    with pytest.raises(ValueError):
        gate.approve(action.id)  # cannot execute twice


def test_gate_executor_failure_is_visible(store):
    def boom(payload):
        raise RuntimeError("platform said no")

    gate = ActionGate(store, {ActionKind.START_CALL: boom})
    action = gate.request(ActionKind.START_CALL, "Call x", {"to": "+1", "say": "hi"})
    failed = gate.approve(action.id)
    assert failed.status.value == "failed"
    assert failed.error == "platform said no"


def test_gate_unknown_action(store):
    gate = ActionGate(store, {})
    with pytest.raises(KeyError):
        gate.approve("nope")


# --- durable jobs ---


def test_jobs_survive_restart_and_interrupted_jobs_requeue(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    provider = StaticProvider(
        default=ProviderResult(
            summary="s [1]",
            claims=[Claim(text="s", source_indexes=[0])],
            sources=[Source(title="t", url="https://example.com")],
        )
    )

    store1 = Store(db)
    runner1 = JobRunner(ResearchPipeline(provider, store1), store1)
    job = runner1.submit("durable question")
    runner1.shutdown()
    assert store1.get_job(job.id).status == JobStatus.DONE
    answer_id = store1.get_job(job.id).answer_id
    store1.close()

    # "Restart": new store and runner over the same file
    store2 = Store(db)
    runner2 = JobRunner(ResearchPipeline(provider, store2), store2)
    recovered = runner2.get(job.id)
    assert recovered.status == JobStatus.DONE
    assert store2.get_answer(answer_id).question == "durable question"
    runner2.shutdown()
    store2.close()


def test_interrupted_job_is_requeued(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store1 = Store(db)
    # simulate a crash: a job left in "running" state
    from jarvis.schemas import ResearchJob

    stuck = ResearchJob(question="stuck", status=JobStatus.RUNNING)
    store1.save_job(stuck)
    store1.close()

    provider = StaticProvider(
        default=ProviderResult(
            summary="s [1]",
            claims=[Claim(text="s", source_indexes=[0])],
            sources=[Source(title="t", url="https://example.com")],
        )
    )
    store2 = Store(db)
    runner = JobRunner(ResearchPipeline(provider, store2), store2)
    runner.shutdown()
    assert store2.get_job(stuck.id).status == JobStatus.DONE
    store2.close()


# --- citation validation ---


def test_out_of_range_citation_demotes_claim(store):
    result = ProviderResult(
        claims=[Claim(text="bad ref", source_indexes=[5])],
        sources=[Source(title="t", url="https://example.com")],
    )
    answer = ResearchPipeline(StaticProvider(default=result), store).ask("q")
    assert answer.status == AnswerStatus.UNCERTAIN
    assert answer.unsupported_claims == ["bad ref"]


def test_non_url_sources_are_dropped_and_reindexed(store):
    result = ProviderResult(
        claims=[
            Claim(text="kept", source_indexes=[1]),
            Claim(text="lost its source", source_indexes=[0]),
        ],
        sources=[
            Source(title="junk", url="not-a-url"),
            Source(title="real", url="https://example.com"),
        ],
    )
    answer = ResearchPipeline(StaticProvider(default=result), store).ask("q")
    assert len(answer.sources) == 1
    assert answer.claims[0].text == "kept"
    assert answer.claims[0].source_indexes == [0]  # remapped
    assert answer.unsupported_claims == ["lost its source"]


# --- scheduler duplicate prevention ---


def test_scheduler_cannot_double_publish(store):
    from datetime import datetime, timedelta, timezone

    from jarvis.scheduler import PublishScheduler
    from jarvis.schemas import DraftStatus

    calls = []

    class CountingConnector:
        name = "x"

        def publish_draft(self, draft):
            calls.append(draft.id)
            return {"id": "posted-1"}

    scheduler = PublishScheduler(store, {"x": CountingConnector()})
    draft = scheduler.create_draft(Platform.X, "hello")
    scheduler.schedule(draft.id, datetime.now(timezone.utc) - timedelta(seconds=1))

    first = scheduler.publish_due()
    second = scheduler.publish_due()  # overlapping/again: must not repost
    assert first[0].status == DraftStatus.PUBLISHED
    assert second == []
    assert calls == [draft.id]
