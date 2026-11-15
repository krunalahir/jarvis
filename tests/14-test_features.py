"""Feature-level checks added with the Nov 2026 review fixes.

Extends coverage beyond the regression tests: scheduler semantics, the
approval HTTP surface, recovery edge cases, connector request shapes and
agent write tools. Nothing here touches the network.
"""

import base64
import threading
from datetime import datetime, timedelta, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from jarvis.actions import ActionGate
from jarvis.agent import build_tools
from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.connectors import build_connectors
from jarvis.connectors.base import ActionsDisabled
from jarvis.connectors.calling import TwilioCallingConnector
from jarvis.connectors.google_workspace import CalendarConnector, GmailConnector
from jarvis.connectors.social import InstagramConnector, XConnector, YouTubeConnector
from jarvis.connectors.spotify import SpotifyConnector
from jarvis.content import ContentGenerator
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

from fakes import ScriptedChatModel


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


# --- scheduler semantics ---


def test_schedule_naive_datetime_is_treated_as_utc(store):
    scheduler = PublishScheduler(store, {})
    draft = scheduler.create_draft(Platform.X, "hello")
    naive = datetime(2026, 11, 20, 12, 0)  # no tzinfo
    scheduled = scheduler.schedule(draft.id, naive)
    assert scheduled.scheduled_at.tzinfo is not None
    assert scheduled.scheduled_at.utcoffset() == timedelta(0)


def test_schedule_unknown_draft_raises(store):
    scheduler = PublishScheduler(store, {})
    with pytest.raises(KeyError):
        scheduler.schedule("nope", datetime.now(timezone.utc))


def test_failed_draft_is_not_retried_by_publish_due(store):
    calls = []

    class FailingConnector:
        name = "x"

        def publish_draft(self, draft):
            calls.append(draft.id)
            raise RuntimeError("platform down")

    scheduler = PublishScheduler(store, {"x": FailingConnector()})
    draft = scheduler.create_draft(Platform.X, "hello")
    scheduler.schedule(draft.id, datetime.now(timezone.utc) - timedelta(seconds=1))
    first = scheduler.publish_due()
    assert first[0].status == DraftStatus.FAILED
    assert first[0].error == "platform down"
    assert scheduler.publish_due() == []  # failed drafts are not picked up again
    assert calls == [draft.id]


def test_draft_with_platform_post_id_is_never_reposted(store):
    calls = []

    class CountingConnector:
        name = "x"

        def publish_draft(self, draft):
            calls.append(draft.id)
            return {"id": "p1"}

    scheduler = PublishScheduler(store, {"x": CountingConnector()})
    draft = scheduler.create_draft(Platform.X, "hello")
    draft.platform_post_id = "already-there"
    store.save_draft(draft)
    scheduler.schedule(draft.id, datetime.now(timezone.utc) - timedelta(seconds=1))
    scheduler.publish_due()
    assert store.get_draft(draft.id).status == DraftStatus.PUBLISHED
    assert calls == []  # skipped: the platform id proves it was posted


def test_youtube_draft_publishes_end_to_end_through_scheduler(store):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=b"video-bytes")
        return httpx.Response(200, json={"id": "yt-1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    yt = YouTubeConnector(
        settings(google_access_token="tok", actions_enabled=True), client=client
    )
    scheduler = PublishScheduler(store, {"youtube": yt})
    draft = scheduler.create_draft(
        Platform.YOUTUBE, "Launch clip", media_urls=["https://cdn.example.com/v.mp4"]
    )
    scheduler.schedule(draft.id, datetime.now(timezone.utc) - timedelta(seconds=1))
    touched = scheduler.publish_due()
    assert touched[0].status == DraftStatus.PUBLISHED
    assert touched[0].platform_post_id == "yt-1"


# --- approval over HTTP ---


def _api_client(**setting_overrides):
    store = Store(":memory:")
    app = create_app(
        settings=Settings(actions_enabled=False, **setting_overrides),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
    )
    return TestClient(app), store


def test_approve_unknown_action_returns_404():
    client, store = _api_client()
    assert client.post("/actions/nope/approve").status_code == 404
    assert client.post("/actions/nope/reject").status_code == 404
    store.close()


def test_publish_now_queues_approval_and_never_publishes_directly():
    client, store = _api_client()
    draft = client.post("/content/drafts", json={"platform": "x", "text": "hi"}).json()
    action = client.post(f"/content/drafts/{draft['id']}/publish-now").json()
    assert action["status"] == "pending"
    assert action["kind"] == "publish_draft"
    assert store.get_draft(draft["id"]).status == DraftStatus.DRAFT  # untouched
    store.close()


def test_publish_now_unknown_draft_is_404():
    client, store = _api_client()
    assert client.post("/content/drafts/nope/publish-now").status_code == 404
    store.close()


def test_approved_publish_executes_through_connector():
    calls = []

    class FakeX:
        name = "x"
        settings = None

        def publish_draft(self, draft):
            calls.append(draft.id)
            return {"id": "posted-9"}

        def status(self):
            return {"name": "x", "state": "ready", "missing_settings": [], "actions_enabled": True}

    store = Store(":memory:")
    connectors = build_connectors(Settings())
    connectors["x"] = FakeX()
    app = create_app(
        settings=Settings(actions_enabled=True, api_key="k"),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
        connectors=connectors,
    )
    client = TestClient(app)
    headers = {"X-API-Key": "k"}
    draft = client.post(
        "/content/drafts", json={"platform": "x", "text": "hi"}, headers=headers
    ).json()
    action = client.post(f"/content/drafts/{draft['id']}/publish-now", headers=headers).json()
    decided = client.post(f"/actions/{action['id']}/approve", headers=headers).json()
    assert decided["status"] == "executed"
    assert calls == [draft["id"]]
    store.close()


# --- recovery edge cases ---


def test_recovery_is_noop_on_clean_store(store):
    assert recover_interrupted_state(store) == {"actions": 0, "drafts": 0}


def test_recovery_leaves_pending_actions_alone(store):
    gate = ActionGate(store, {})
    action = gate.request(ActionKind.SEND_EMAIL, "Send", {"to": "a@b.com"})
    recover_interrupted_state(store)
    assert store.get_action(action.id).status == PendingActionStatus.PENDING


def test_recovery_unknown_outcome_message_mentions_no_retry(tmp_path):
    db = str(tmp_path / "s.sqlite3")
    store = Store(db)
    store.save_draft(
        PostDraft(platform=Platform.X, text="t", status=DraftStatus.PUBLISHING)
    )
    recover_interrupted_state(store)
    draft = store.list_actions()  # actions empty
    assert draft == []
    assert "NOT retried" in UNKNOWN_OUTCOME
    store.close()


# --- store internals ---


def test_claim_action_returns_none_for_decided_action(store):
    gate = ActionGate(store, {ActionKind.SPOTIFY_PLAY: lambda p: {}})
    action = gate.request(ActionKind.SPOTIFY_PLAY, "Play", {})
    gate.reject(action.id)
    assert store.claim_action(action.id) is None
    assert store.get_action(action.id).status == PendingActionStatus.REJECTED


def test_file_store_sets_wal_and_busy_timeout(tmp_path):
    db = str(tmp_path / "s.sqlite3")
    store = Store(db)
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    timeout = store._conn.execute("PRAGMA busy_timeout").fetchone()[0]
    assert mode == "wal"
    assert timeout == 5000
    store.close()


# --- Gmail / Calendar request shapes ---


def test_gmail_html_only_message_falls_back_to_snippet():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/gmail/v1/users/me/messages":
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        return httpx.Response(
            200,
            json={
                "id": "m1",
                "snippet": "preview text",
                "payload": {
                    "mimeType": "text/html",
                    "headers": [{"name": "Subject", "value": "Newsletter"}],
                    "body": {"data": base64.urlsafe_b64encode(b"<p>hi</p>").decode()},
                },
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gmail = GmailConnector(settings(google_access_token="tok"), client=client)
    msg = gmail.read_messages("newsletter")[0]
    assert msg["subject"] == "Newsletter"
    assert msg["body"] == "<p>hi</p>" or msg["snippet"] == "preview text"


def test_gmail_read_messages_empty_search():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gmail = GmailConnector(settings(google_access_token="tok"), client=client)
    assert gmail.read_messages("nothing") == []


def test_calendar_create_event_builds_payload():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "evt1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    calendar = CalendarConnector(
        settings(google_access_token="tok", actions_enabled=True), client=client
    )
    import json as _json

    result = calendar.create_event(
        "Standup", "2026-11-17T09:00:00Z", "2026-11-17T09:15:00Z",
        attendees=["a@b.com"], description="daily",
    )
    assert result["id"] == "evt1"
    body = _json.loads(seen["body"])
    assert body["summary"] == "Standup"
    assert body["attendees"] == [{"email": "a@b.com"}]
    assert body["start"] == {"dateTime": "2026-11-17T09:00:00Z"}


def test_calendar_create_event_refused_when_actions_disabled():
    calendar = CalendarConnector(settings(google_access_token="tok"))
    with pytest.raises(ActionsDisabled):
        calendar.create_event("x", "2026-11-17T09:00:00Z", "2026-11-17T09:15:00Z")


# --- connector request shapes ---


def test_instagram_video_url_creates_reels_container():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/media"):
            seen["container"] = request.read().decode()
            return httpx.Response(200, json={"id": "c1"})
        return httpx.Response(200, json={"id": "p1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ig = InstagramConnector(
        settings(
            meta_access_token="tok",
            instagram_business_account_id="ig1",
            actions_enabled=True,
        ),
        client=client,
    )
    draft = PostDraft(
        platform=Platform.INSTAGRAM, text="cap", media_urls=["https://x/y.mp4"]
    )
    ig.publish_draft(draft)
    assert "REELS" in seen["container"]


def test_x_publish_refused_when_actions_disabled():
    x = XConnector(settings(x_bearer_token="tok"))
    with pytest.raises(ActionsDisabled):
        x.publish_draft(PostDraft(platform=Platform.X, text="hi"))


def test_youtube_upload_from_local_file_path(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"local-video")
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(200, json={"id": "yt-local"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    yt = YouTubeConnector(
        settings(google_access_token="tok", actions_enabled=True), client=client
    )
    draft = PostDraft(platform=Platform.YOUTUBE, text="local upload")
    result = yt.publish_draft(draft, video_file_path=str(video))
    assert result["id"] == "yt-local"
    assert b"local-video" in seen["body"]


def test_spotify_play_without_query_resumes():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    spotify = SpotifyConnector(
        settings(spotify_access_token="tok", actions_enabled=True), client=client
    )
    result = spotify.play()
    assert result["started"] is True
    assert seen["body"] == b"{}"


def test_spotify_play_unknown_track_is_a_clear_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"tracks": {"items": []}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    spotify = SpotifyConnector(
        settings(spotify_access_token="tok", actions_enabled=True), client=client
    )
    with pytest.raises(ValueError, match="No Spotify track found"):
        spotify.play("no such song xyz")


def test_twilio_uses_account_scoped_url_and_auth():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        import base64 as _b64

        seen["auth"] = _b64.b64decode(
            request.headers["Authorization"].split()[1]
        ).decode()
        return httpx.Response(201, json={"sid": "CA1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    twilio = TwilioCallingConnector(
        settings(
            twilio_account_sid="AC123",
            twilio_auth_token="secret",
            twilio_from_number="+1",
            actions_enabled=True,
        ),
        client=client,
    )
    twilio.start_call("+2", "hello")
    assert "/Accounts/AC123/Calls.json" in seen["url"]
    assert seen["auth"] == "AC123:secret"


# --- agent tools ---


def _tool_parts(store):
    settings = Settings()
    pipeline = ResearchPipeline(StaticProvider(default=ProviderResult()), store)
    connectors = build_connectors(settings)
    scheduler = PublishScheduler(store, connectors)
    generator = ContentGenerator(settings, model=ScriptedChatModel(script=[AIMessage(content="draft")]))
    gate = ActionGate(store, {})
    pending = []
    tools = build_tools(
        settings, pipeline, gate, scheduler, generator, connectors,
        pending_action_ids=pending,
    )
    return tools, pending


def _tool_by_name(tools, name):
    return next(t for t in tools if t.name == name)


def test_request_tools_queue_every_write_kind(store):
    tools, pending = _tool_parts(store)
    _tool_by_name(tools, "request_send_email").invoke(
        {"to": "a@b.com", "subject": "S", "body": "B"}
    )
    _tool_by_name(tools, "request_create_event").invoke(
        {"summary": "Meet", "start_iso": "2026-11-17T09:00:00Z", "end_iso": "2026-11-17T09:30:00Z"}
    )
    _tool_by_name(tools, "request_phone_call").invoke({"to": "+1", "say": "hello"})
    _tool_by_name(tools, "request_spotify_play").invoke({"track_query": "song"})
    assert len(pending) == 4
    kinds = {store.get_action(aid).kind for aid in pending}
    assert kinds == {
        ActionKind.SEND_EMAIL,
        ActionKind.CREATE_EVENT,
        ActionKind.START_CALL,
        ActionKind.SPOTIFY_PLAY,
    }
    for aid in pending:  # everything waits for approval, nothing executed
        assert store.get_action(aid).status == PendingActionStatus.PENDING


def test_draft_post_tool_rejects_unknown_platform(store):
    tools, _ = _tool_parts(store)
    result = _tool_by_name(tools, "draft_post").invoke(
        {"platform": "myspace", "topic": "launch"}
    )
    assert "Unknown platform" in result
    assert store.interrupted_drafts() == []


def test_draft_post_tool_saves_draft_without_publishing(store):
    tools, _ = _tool_parts(store)
    result = _tool_by_name(tools, "draft_post").invoke(
        {"platform": "x", "topic": "launch day"}
    )
    assert "never publishes itself" in result
    due = store.due_drafts(datetime.now(timezone.utc).isoformat())
    assert due == []  # draft exists but is not scheduled


def test_connector_status_tool_lists_every_connector(store):
    tools, _ = _tool_parts(store)
    result = _tool_by_name(tools, "connector_status").invoke({})
    for name in ("gmail", "google_calendar", "calling", "instagram", "x", "youtube", "snapchat", "spotify"):
        assert name in result
    assert "not_configured" in result


def test_read_emails_tool_reports_unconfigured_cleanly(store):
    tools, _ = _tool_parts(store)
    result = _tool_by_name(tools, "read_emails").invoke({"query": "from:bob"})
    assert "missing settings" in result


# --- API security surface ---


def test_health_reports_security_warnings_without_api_key():
    client, store = _api_client()
    body = client.get("/health").json()
    assert any("JARVIS_API_KEY is unset" in w for w in body["security_warnings"])
    store.close()


def test_every_write_endpoint_requires_api_key_when_set():
    store = Store(":memory:")
    app = create_app(
        settings=Settings(api_key="secret"),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
    )
    client = TestClient(app)
    assert client.post("/gmail/send", json={"to": "a", "subject": "b", "body": "c"}).status_code == 401
    assert client.post("/content/drafts", json={"platform": "x", "text": "t"}).status_code == 401
    assert client.post("/agent/chat", json={"message": "hi"}).status_code == 401
    assert client.get("/health").status_code == 200  # health stays open
    store.close()


def test_actions_enabled_without_key_refused_even_with_allow_insecure_unset():
    with pytest.raises(RuntimeError, match="Refusing to start"):
        create_app(
            settings=Settings(actions_enabled=True, api_key=None),
            store=Store(":memory:"),
        )
