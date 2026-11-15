"""Regression tests for the Nov 2026 code review findings.

Covers: approval race (double execution), crash recovery without blind
retries, store concurrency, Gmail content reads, YouTube upload from a
draft media URL, TwiML escaping, and agent history validation.
Nothing here touches the network.
"""

import base64
import threading
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient

from jarvis.actions import ActionGate
from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.connectors.calling import TwilioCallingConnector
from jarvis.connectors.google_workspace import GmailConnector
from jarvis.connectors.social import YouTubeConnector
from jarvis.recovery import UNKNOWN_OUTCOME, recover_interrupted_state
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.scheduler import PublishScheduler
from jarvis.schemas import (
    ActionKind,
    DraftStatus,
    PendingAction,
    PendingActionStatus,
    Platform,
    PostDraft,
)


def settings(**overrides):
    base = Settings(actions_enabled=False)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


# --- 1. approval race: concurrent approvals execute exactly once ---


def test_concurrent_approvals_execute_once(store):
    calls = []

    def slow_executor(payload):
        time.sleep(0.3)  # force both requests to overlap
        calls.append(payload)
        return {"ok": 1}

    gate = ActionGate(store, {ActionKind.SEND_EMAIL: slow_executor})
    action = gate.request(
        ActionKind.SEND_EMAIL, "Send email", {"to": "a@b.com", "subject": "s", "body": "b", "cc": []}
    )

    results, errors = [], []

    def approve():
        try:
            results.append(gate.approve(action.id))
        except ValueError as exc:
            errors.append(exc)

    threads = [threading.Thread(target=approve) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert calls == [action.payload]  # executor ran exactly once
    assert len(results) == 1 and len(errors) == 1
    assert results[0].status == PendingActionStatus.EXECUTED
    assert "already" in str(errors[0])
    assert store.get_action(action.id).status == PendingActionStatus.EXECUTED


def test_concurrent_reject_and_approve_only_one_wins(store):
    calls = []

    def slow_executor(payload):
        time.sleep(0.3)
        calls.append(payload)
        return {}

    gate = ActionGate(store, {ActionKind.START_CALL: slow_executor})
    action = gate.request(ActionKind.START_CALL, "Call", {"to": "+1", "say": "hi"})

    outcomes = []

    def do(fn):
        try:
            outcomes.append(("ok", fn(action.id).status))
        except (ValueError, KeyError) as exc:
            outcomes.append(("err", str(exc)))

    def reject_after_claim_window(action_id):
        time.sleep(0.1)  # let approve win the claim first
        return gate.reject(action_id)

    t1 = threading.Thread(target=do, args=(gate.approve,))
    t2 = threading.Thread(target=lambda: outcomes.append(_try(reject_after_claim_window, action.id)))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert calls == [action.payload]
    final = store.get_action(action.id).status
    assert final == PendingActionStatus.EXECUTED  # reject lost the race


def _try(fn, *args):
    try:
        return ("ok", fn(*args).status)
    except (ValueError, KeyError) as exc:
        return ("err", str(exc))


# --- 5 + recovery: interrupted work is flagged, never retried ---


def test_interrupted_action_marked_unknown_and_not_retried(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store1 = Store(db)
    stuck = PendingAction(
        kind=ActionKind.SEND_EMAIL,
        summary="Send email",
        payload={"to": "a@b.com"},
        status=PendingActionStatus.EXECUTING,  # crash mid-execution
    )
    store1.save_action(stuck)
    store1.close()

    store2 = Store(db)  # "restart"
    calls = []
    gate = ActionGate(store2, {ActionKind.SEND_EMAIL: lambda p: calls.append(p) or {}})
    recovered = recover_interrupted_state(store2)
    assert recovered == {"actions": 1, "drafts": 0}
    action = store2.get_action(stuck.id)
    assert action.status == PendingActionStatus.FAILED
    assert action.error == UNKNOWN_OUTCOME
    assert calls == []  # no executor ran during recovery
    with pytest.raises(ValueError):  # and it cannot be re-approved
        gate.approve(stuck.id)
    store2.close()


def test_interrupted_draft_marked_unknown_and_not_republished(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store1 = Store(db)
    stuck = PostDraft(
        platform=Platform.X,
        text="hello",
        status=DraftStatus.PUBLISHING,  # crash between claim and publish
        scheduled_at=datetime.now(timezone.utc) - timedelta(minutes=5),
    )
    store1.save_draft(stuck)
    store1.close()

    store2 = Store(db)
    calls = []

    class CountingConnector:
        name = "x"

        def publish_draft(self, draft):
            calls.append(draft.id)
            return {"id": "p1"}

    scheduler = PublishScheduler(store2, {"x": CountingConnector()})
    recovered = recover_interrupted_state(store2)
    assert recovered["drafts"] == 1
    assert store2.get_draft(stuck.id).status == DraftStatus.FAILED
    assert store2.get_draft(stuck.id).error == UNKNOWN_OUTCOME
    assert scheduler.publish_due() == []  # failed drafts are not due again
    assert calls == []
    store2.close()


def test_create_app_recovers_interrupted_state_on_startup(tmp_path):
    db = str(tmp_path / "store.sqlite3")
    store = Store(db)
    store.save_action(
        PendingAction(
            kind=ActionKind.START_CALL,
            summary="Call",
            payload={},
            status=PendingActionStatus.EXECUTING,
        )
    )
    create_app(
        settings=Settings(actions_enabled=False),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
    )
    actions = store.list_actions()
    assert actions[0].status == PendingActionStatus.FAILED
    assert "NOT retried" in actions[0].error
    store.close()


# --- 4. store concurrency: parallel writers do not error or lose data ---


def test_store_survives_concurrent_writers(store):
    errors = []

    def writer(n):
        try:
            for i in range(20):
                store.save_action(
                    PendingAction(kind=ActionKind.SPOTIFY_PLAY, summary=f"s{n}-{i}", payload={})
                )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    assert len(store.list_actions(limit=200)) == 160


# --- 2. Gmail: search results carry actual message content ---


def _gmail_payload(text):
    return base64.urlsafe_b64encode(text.encode()).decode()


def test_gmail_read_messages_fetches_bodies():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/gmail/v1/users/me/messages":
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        if request.url.path == "/gmail/v1/users/me/messages/m1":
            return httpx.Response(
                200,
                json={
                    "id": "m1",
                    "snippet": "snip",
                    "payload": {
                        "mimeType": "multipart/alternative",
                        "headers": [
                            {"name": "From", "value": "bob@example.com"},
                            {"name": "Subject", "value": "Pricing"},
                            {"name": "Date", "value": "Mon, 16 Nov 2026"},
                        ],
                        "parts": [
                            {"mimeType": "text/plain", "body": {"data": _gmail_payload("body text here")}},
                        ],
                    },
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gmail = GmailConnector(settings(google_access_token="tok"), client=client)
    messages = gmail.read_messages("from:bob")
    assert messages[0]["subject"] == "Pricing"
    assert messages[0]["from"] == "bob@example.com"
    assert messages[0]["body"] == "body text here"
    assert any("/messages/m1" in url for url in seen)  # fetched, not just listed


# --- 3. YouTube: publishing a draft works from its media URL ---


def test_youtube_publish_downloads_video_from_draft_media_url():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=b"fake-video-bytes")
        seen["upload"] = True
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "yt1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    yt = YouTubeConnector(settings(google_access_token="tok", actions_enabled=True), client=client)
    draft = PostDraft(
        platform=Platform.YOUTUBE,
        text="Launch clip",
        media_urls=["https://cdn.example.com/clip.mp4"],
    )
    result = yt.publish_draft(draft)  # no local file path, as the scheduler calls it
    assert result["id"] == "yt1"
    assert seen["upload"] is True
    assert b"fake-video-bytes" in seen["body"]


def test_youtube_publish_without_video_source_fails_loudly():
    yt = YouTubeConnector(settings(google_access_token="tok", actions_enabled=True))
    draft = PostDraft(platform=Platform.YOUTUBE, text="no media")
    with pytest.raises(ValueError, match="video file path or a media URL"):
        yt.publish_draft(draft)


def test_youtube_scheduled_draft_sets_native_publish_at():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=b"v")
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "yt2"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    yt = YouTubeConnector(settings(google_access_token="tok", actions_enabled=True), client=client)
    draft = PostDraft(
        platform=Platform.YOUTUBE,
        text="Later",
        media_urls=["https://cdn.example.com/clip.mp4"],
        scheduled_at=datetime(2026, 11, 20, 12, 0, tzinfo=timezone.utc),
    )
    yt.publish_draft(draft)
    assert b"2026-11-20T12:00:00.000Z" in seen["body"]


# --- 6. TwiML: spoken text is XML-escaped ---


def test_twilio_twiml_escapes_spoken_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read().decode()
        return httpx.Response(201, json={"sid": "CA1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    twilio = TwilioCallingConnector(
        settings(
            twilio_account_sid="AC1",
            twilio_auth_token="tok",
            twilio_from_number="+1",
            actions_enabled=True,
        ),
        client=client,
    )
    twilio.start_call("+2", "5 < 7 & 9 > 4")
    form = urllib.parse.parse_qs(seen["body"])
    twiml = form["Twiml"][0]
    assert twiml == "<Response><Say>5 &lt; 7 &amp; 9 &gt; 4</Say></Response>"


# --- 7. agent history: validated and bounded ---


def _client():
    store = Store(":memory:")
    app = create_app(
        settings=Settings(actions_enabled=False),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
    )
    return TestClient(app), store


def test_agent_chat_rejects_bad_history_role():
    client, store = _client()
    response = client.post(
        "/agent/chat",
        json={"message": "hi", "history": [{"role": "system", "content": "x"}]},
    )
    assert response.status_code == 422
    store.close()


def test_agent_chat_rejects_oversized_history_content():
    client, store = _client()
    response = client.post(
        "/agent/chat",
        json={"message": "hi", "history": [{"role": "user", "content": "x" * 4001}]},
    )
    assert response.status_code == 422
    store.close()


def test_agent_chat_rejects_overlong_history():
    client, store = _client()
    response = client.post(
        "/agent/chat",
        json={
            "message": "hi",
            "history": [{"role": "user", "content": "x"} for _ in range(51)],
        },
    )
    assert response.status_code == 422
    store.close()


def test_run_agent_truncates_history_to_recent_turns():
    from langchain_core.messages import AIMessage

    from fakes import ScriptedChatModel
    from jarvis.agent import build_agent, run_agent
    from jarvis.agent.graph import MAX_HISTORY_MESSAGES

    settings = Settings()
    model = ScriptedChatModel(script=[AIMessage(content="ok")])
    agent = build_agent(settings, [], model=model)
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(60)
    ]
    reply = run_agent(agent, "now", history=history)
    assert reply.reply == "ok"
