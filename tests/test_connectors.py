import httpx
import pytest

from jarvis.config import Settings
from jarvis.connectors import (
    ActionsDisabled,
    ConnectorNotConfigured,
    PlatformAccessRequired,
    build_connectors,
)
from jarvis.connectors.social import InstagramConnector, XConnector
from jarvis.connectors.spotify import SpotifyConnector
from jarvis.connectors.calling import TwilioCallingConnector
from jarvis.connectors.google_workspace import GmailConnector
from jarvis.schemas import Platform, PostDraft

from google_fakes import FakeGoogleSignIn


def settings(**overrides):
    base = Settings(actions_enabled=False)
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_all_connectors_report_not_configured_by_default():
    connectors = build_connectors(Settings())
    states = {c.name: c.status()["state"] for c in connectors.values()}
    assert states["gmail"] == "not_configured"
    assert states["spotify"] == "not_configured"
    assert states["x"] == "not_configured"


def test_write_refused_when_actions_disabled():
    gmail = GmailConnector(settings(), google_oauth=FakeGoogleSignIn())
    assert gmail.status()["state"] == "configured_actions_disabled"
    with pytest.raises(ActionsDisabled):
        gmail.send_email("a@b.com", "hi", "body")


def test_write_refused_when_not_configured():
    gmail = GmailConnector(settings(actions_enabled=True))
    with pytest.raises(ConnectorNotConfigured):
        gmail.send_email("a@b.com", "hi", "body")


def test_gmail_send_builds_api_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"id": "msg1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    gmail = GmailConnector(
        settings(actions_enabled=True),
        client=client,
        google_oauth=FakeGoogleSignIn("tok"),
    )
    result = gmail.send_email("a@b.com", "Subject", "Body")
    assert result["id"] == "msg1"
    assert seen["url"].endswith("/messages/send")
    assert seen["auth"] == "Bearer tok"


def test_x_publish_posts_text():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.read()
        return httpx.Response(201, json={"data": {"id": "t1"}})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    x = XConnector(settings(x_bearer_token="tok", actions_enabled=True), client=client)
    draft = PostDraft(platform=Platform.X, text="hello world")
    result = x.publish_draft(draft)
    assert result["data"]["id"] == "t1"
    assert b"hello world" in seen["body"]


def test_instagram_two_step_publish():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if request.url.path.endswith("/media"):
            return httpx.Response(200, json={"id": "container1"})
        return httpx.Response(200, json={"id": "post1"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    ig = InstagramConnector(
        settings(
            meta_access_token="tok",
            instagram_business_account_id="ig1",
            actions_enabled=True,
        ),
        client=client,
    )
    draft = PostDraft(platform=Platform.INSTAGRAM, text="cap", media_urls=["https://x/y.jpg"])
    result = ig.publish_draft(draft)
    assert result["id"] == "post1"
    assert calls[0].endswith("/ig1/media")
    assert calls[1].endswith("/ig1/media_publish")


def test_spotify_search_is_read_and_play_is_write():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/search":
            return httpx.Response(
                200,
                json={"tracks": {"items": [{"uri": "spotify:track:1", "name": "Song", "artists": [{"name": "Artist"}]}]}},
            )
        return httpx.Response(204)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    configured_only = SpotifyConnector(settings(spotify_access_token="tok"), client=client)
    assert configured_only.search_track("song")[0]["name"] == "Song"
    with pytest.raises(ActionsDisabled):
        configured_only.play("song")

    enabled = SpotifyConnector(
        settings(spotify_access_token="tok", actions_enabled=True), client=client
    )
    result = enabled.play("song")
    assert result["started"] is True
    assert result["body"]["uris"] == ["spotify:track:1"]


def test_twilio_call_gated():
    twilio = TwilioCallingConnector(
        settings(
            twilio_account_sid="AC1",
            twilio_auth_token="tok",
            twilio_from_number="+1",
        )
    )
    with pytest.raises(ActionsDisabled):
        twilio.start_call("+2", "hello")


def test_snapchat_scaffold_fails_loudly():
    from jarvis.connectors.social import SnapchatConnector

    snap = SnapchatConnector(Settings())
    with pytest.raises(PlatformAccessRequired):
        snap.publish_draft(PostDraft(platform=Platform.SNAPCHAT, text="hi"))


def test_gmail_search_and_calendar_list_are_reads():
    from jarvis.connectors.google_workspace import CalendarConnector

    def handler(request: httpx.Request) -> httpx.Response:
        if "messages" in request.url.path:
            return httpx.Response(200, json={"messages": [{"id": "m1"}]})
        return httpx.Response(200, json={"items": [{"id": "e1", "summary": "Meet"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    s = settings()  # signed in to Google, actions disabled
    google = FakeGoogleSignIn("tok")
    gmail = GmailConnector(s, client=client, google_oauth=google)
    assert gmail.search_messages("from:bob") == [{"id": "m1"}]
    calendar = CalendarConnector(s, client=client, google_oauth=google)
    events = calendar.list_events("2026-11-14T00:00:00Z")
    assert events[0]["summary"] == "Meet"
