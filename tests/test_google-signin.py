"""Google sign-in: started from chat, finished in the browser, then used
by Gmail, Calendar and YouTube.

All offline. The Google libraries are replaced by fakes (google_fakes.py),
the model is scripted, and HTTP goes through mock transports.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolCall

from jarvis import google_oauth
from jarvis.actions import ActionGate
from jarvis.agent import build_agent, build_tools
from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.connectors import ConnectorNotConfigured, build_connectors
from jarvis.connectors.google_workspace import GmailConnector
from jarvis.content import ContentGenerator
from jarvis.google_oauth import (
    FEATURE_SCOPES,
    GMAIL_SEND,
    SCOPES,
    GoogleOAuth,
    GoogleSignInError,
)
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.scheduler import PublishScheduler

from fakes import ScriptedChatModel
from google_fakes import GOOD_CODE, FakeCredentials, FakeFlow


@pytest.fixture
def google_libs(monkeypatch):
    """Swap the Google libraries for fakes (installed or not)."""
    monkeypatch.setattr(google_oauth, "Flow", FakeFlow)
    monkeypatch.setattr(google_oauth, "Credentials", FakeCredentials)
    monkeypatch.setattr(google_oauth, "Request", lambda: None)
    FakeFlow.reset()


@pytest.fixture
def settings(google_libs, tmp_path):
    """Sign-in is fully set up: fake libraries plus a client id and secret."""
    return Settings(
        google_client_id="client-id",
        google_client_secret="client-secret",
        google_token_file=str(tmp_path / "google_tokens.json"),
    )


def state_of(link: str) -> str:
    return parse_qs(urlsplit(link).query)["state"][0]


def sign_in(oauth: GoogleOAuth) -> dict:
    """Do what the user does: get a link, press Allow, come back."""
    return oauth.finish(GOOD_CODE, state_of(oauth.start()))


def make_tools(settings, oauth, links=None, pending=None) -> dict:
    store = Store(":memory:")
    connectors = build_connectors(settings, google_oauth=oauth)
    tools = build_tools(
        settings,
        ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        ActionGate(store, {}),
        PublishScheduler(store, connectors),
        ContentGenerator(settings),
        connectors,
        pending_action_ids=pending,
        google_oauth=oauth,
        sign_in_links=links,
    )
    return {t.name: t for t in tools}


def chat_client(settings, script, **app_kwargs) -> TestClient:
    model = ScriptedChatModel(script=script)
    app = create_app(
        settings=settings,
        store=Store(":memory:"),
        agent_builder=lambda tools: build_agent(settings, tools, model=model),
        **app_kwargs,
    )
    return TestClient(app)


def connector_states(client, headers=None) -> dict:
    listing = client.get("/connectors/status", headers=headers).json()
    return {c["name"]: c["state"] for c in listing}


# --- the sign-in itself ---


def test_sign_in_needs_client_id_and_secret(google_libs):
    oauth = GoogleOAuth(Settings())
    assert "GOOGLE_CLIENT_ID" in oauth.problem()
    assert "GOOGLE_CLIENT_SECRET" in oauth.problem()
    with pytest.raises(GoogleSignInError):
        oauth.start()
    assert oauth.connected() is False


def test_a_missing_google_library_is_reported_not_a_crash(settings, monkeypatch):
    monkeypatch.setattr(google_oauth, "Flow", None)
    oauth = GoogleOAuth(settings)
    assert "not installed" in oauth.problem()
    with pytest.raises(GoogleSignInError, match="not installed"):
        oauth.start()


def test_full_sign_in_saves_the_token_and_knows_the_account(settings):
    oauth = GoogleOAuth(settings)
    assert oauth.problem() is None
    link = oauth.start()
    assert link.startswith("https://accounts.google.com/")
    assert oauth.connected() is False  # a link alone connects nothing

    status = oauth.finish(GOOD_CODE, state_of(link))

    assert status["connected"] is True
    assert status["email"] == "user@example.com"
    assert status["missing_scopes"] == []
    assert oauth.access_token() == "access-1"
    saved = json.loads(oauth.token_file.read_text())
    assert saved["refresh_token"] == "refresh-1"
    if os.name == "posix":  # owner-only; Windows has no such mode bits
        assert oauth.token_file.stat().st_mode & 0o777 == 0o600


def test_redirect_uri_and_code_verifier_reach_the_code_exchange(settings):
    oauth = GoogleOAuth(settings)
    link = oauth.start()
    oauth.finish(GOOD_CODE, state_of(link))
    exchange = FakeFlow.exchanges[-1]
    assert exchange["redirect_uri"] == settings.google_redirect_uri
    assert exchange["code_verifier"] == f"verifier-for-{state_of(link)}"


def test_a_link_works_once_and_cannot_be_guessed(settings):
    oauth = GoogleOAuth(settings)
    with pytest.raises(GoogleSignInError):
        oauth.finish(GOOD_CODE, "made-up-state")
    state = state_of(oauth.start())
    oauth.finish(GOOD_CODE, state)
    with pytest.raises(GoogleSignInError):
        oauth.finish(GOOD_CODE, state)  # already used


def test_an_expired_link_is_refused(settings, monkeypatch):
    monkeypatch.setattr(google_oauth, "SIGN_IN_LINK_TTL_SECONDS", -1)
    oauth = GoogleOAuth(settings)
    with pytest.raises(GoogleSignInError, match="expired"):
        oauth.finish(GOOD_CODE, state_of(oauth.start()))
    assert oauth.connected() is False


def test_a_code_google_rejects_connects_nothing(settings):
    oauth = GoogleOAuth(settings)
    with pytest.raises(RuntimeError, match="invalid_grant"):
        oauth.finish("wrong-code", state_of(oauth.start()))
    assert oauth.connected() is False


def test_an_expired_access_token_is_renewed_and_saved(settings):
    oauth = GoogleOAuth(settings)
    sign_in(oauth)
    saved = json.loads(oauth.token_file.read_text())
    long_ago = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(hours=2)
    saved["expiry"] = long_ago.isoformat() + "Z"
    oauth.token_file.write_text(json.dumps(saved))

    assert oauth.access_token() == "renewed-1"
    assert oauth.access_token() == "renewed-1"  # saved, so not renewed again
    assert FakeCredentials.refresh_calls == 1
    after = json.loads(oauth.token_file.read_text())
    assert after["email"] == "user@example.com"  # kept across the renewal
    assert after["scopes"] == saved["scopes"]


def test_an_unticked_permission_is_named_not_a_bare_403(settings):
    FakeFlow.granted = [s for s in SCOPES if s != GMAIL_SEND]
    oauth = GoogleOAuth(settings)
    status = sign_in(oauth)
    assert status["missing_scopes"] == [GMAIL_SEND]

    settings.actions_enabled = True
    gmail = GmailConnector(settings, google_oauth=oauth)
    assert "not granted" in gmail.status()["note"]
    with pytest.raises(ConnectorNotConfigured, match="permission"):
        gmail.send_email("a@b.com", "hi", "body")


def test_only_feature_permissions_are_ever_reported_missing(settings):
    # Whatever Google echoes back for the identity scopes (openid, email),
    # it must never look like a missing permission.
    FakeFlow.granted = list(FEATURE_SCOPES)
    oauth = GoogleOAuth(settings)
    assert sign_in(oauth)["missing_scopes"] == []


# --- connectors use the sign-in, not an environment token ---


def test_gmail_calendar_and_youtube_follow_the_sign_in(settings):
    oauth = GoogleOAuth(settings)
    connectors = build_connectors(settings, google_oauth=oauth)
    for name in ("gmail", "google_calendar", "youtube"):
        assert connectors[name].status()["state"] == "not_configured"

    sign_in(oauth)

    for name in ("gmail", "google_calendar", "youtube"):
        assert connectors[name].status()["state"] == "configured_actions_disabled"
    assert connectors["spotify"].status()["state"] == "not_configured"  # untouched


def test_gmail_sends_the_token_from_the_sign_in(settings):
    seen = {}

    def gmail_api(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"messages": [{"id": "m1"}]})

    oauth = GoogleOAuth(settings)
    sign_in(oauth)
    client = httpx.Client(transport=httpx.MockTransport(gmail_api))
    gmail = GmailConnector(settings, client=client, google_oauth=oauth)
    assert gmail.search_messages("from:bob") == [{"id": "m1"}]
    assert seen["auth"] == "Bearer access-1"


def test_a_google_api_error_keeps_googles_explanation(settings):
    from jarvis.connectors import ConnectorError

    def gmail_api(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403, json={"error": {"code": 403, "message": "Gmail API is disabled."}}
        )

    oauth = GoogleOAuth(settings)
    sign_in(oauth)
    client = httpx.Client(transport=httpx.MockTransport(gmail_api))
    gmail = GmailConnector(settings, client=client, google_oauth=oauth)
    with pytest.raises(ConnectorError, match="Gmail API is disabled"):
        gmail.search_messages("anything")


# --- the agent's tools ---


def test_the_agent_is_given_the_connect_google_tool(settings):
    tools = make_tools(settings, GoogleOAuth(settings))
    assert "connect_google" in tools


def test_connect_google_hands_out_a_link_and_records_it(settings):
    links = []
    oauth = GoogleOAuth(settings)
    tools = make_tools(settings, oauth, links=links)

    answer = tools["connect_google"].invoke({})

    assert len(links) == 1
    assert links[0] in answer
    assert "NOT connected yet" in answer
    assert oauth.connected() is False


def test_connect_google_says_so_when_already_connected(settings):
    links = []
    oauth = GoogleOAuth(settings)
    sign_in(oauth)
    tools = make_tools(settings, oauth, links=links)

    answer = tools["connect_google"].invoke({})
    assert "already connected as user@example.com" in answer
    assert links == []

    again = tools["connect_google"].invoke({"reconnect": True})
    assert len(links) == 1 and links[0] in again


def test_connect_google_explains_missing_setup(google_libs):
    oauth = GoogleOAuth(Settings())
    links = []
    answer = make_tools(Settings(), oauth, links=links)["connect_google"].invoke({})
    assert "not set up yet" in answer
    assert "GOOGLE_CLIENT_ID" in answer
    assert links == []


def test_reading_mail_before_sign_in_points_to_connect_google(settings):
    tools = make_tools(settings, GoogleOAuth(settings))
    answer = tools["read_emails"].invoke({"query": "from:bob"})
    assert "Google is not connected" in answer
    assert "connect_google" in answer


def test_queuing_an_email_before_sign_in_still_queues_and_warns(settings):
    pending = []
    tools = make_tools(settings, GoogleOAuth(settings), pending=pending)
    answer = tools["request_send_email"].invoke(
        {"to": "a@b.com", "subject": "Hi", "body": "Hello"}
    )
    assert len(pending) == 1
    assert "Google is not connected yet" in answer


def test_queuing_an_email_after_sign_in_has_no_warning(settings):
    oauth = GoogleOAuth(settings)
    sign_in(oauth)
    tools = make_tools(settings, oauth)
    answer = tools["request_send_email"].invoke(
        {"to": "a@b.com", "subject": "Hi", "body": "Hello"}
    )
    assert "not connected" not in answer


# --- end to end through the HTTP API ---


def test_sign_in_starts_in_chat_and_finishes_in_the_browser(settings):
    script = [
        AIMessage(content="", tool_calls=[ToolCall(name="connect_google", args={}, id="t1")]),
        AIMessage(content="Open the link I sent you to connect Google."),
    ]
    client = chat_client(settings, script)

    body = client.post("/agent/chat", json={"message": "connect my google"}).json()
    assert body["status"] == "answered"
    assert body["steps"][0]["tool"] == "connect_google"
    link = body["google_sign_in_url"]  # the exact link, not the model's copy
    assert link.startswith("https://accounts.google.com/")
    assert connector_states(client)["gmail"] == "not_configured"

    # The user presses Allow; Google sends their browser here.
    done = client.get(
        "/auth/google/callback", params={"code": GOOD_CODE, "state": state_of(link)}
    )
    assert done.status_code == 200
    assert "Google is connected" in done.text
    assert "user@example.com" in done.text

    states = connector_states(client)
    assert states["gmail"] == "configured_actions_disabled"
    assert states["google_calendar"] == "configured_actions_disabled"
    assert states["youtube"] == "configured_actions_disabled"


def test_a_chat_turn_without_sign_in_has_no_link(settings):
    client = chat_client(settings, [AIMessage(content="Hello.")])
    body = client.post("/agent/chat", json={"message": "hi"}).json()
    assert body["google_sign_in_url"] is None


def test_the_callback_refuses_a_link_it_did_not_issue(settings):
    client = chat_client(settings, [AIMessage(content="hi")])
    bad = client.get(
        "/auth/google/callback", params={"code": GOOD_CODE, "state": "made-up"}
    )
    assert bad.status_code == 400
    assert "Google was not connected" in bad.text
    assert connector_states(client)["gmail"] == "not_configured"


def test_the_callback_reports_a_code_google_rejects(settings):
    script = [
        AIMessage(content="", tool_calls=[ToolCall(name="connect_google", args={}, id="t1")]),
        AIMessage(content="Open the link."),
    ]
    client = chat_client(settings, script)
    link = client.post("/agent/chat", json={"message": "connect google"}).json()[
        "google_sign_in_url"
    ]
    refused = client.get(
        "/auth/google/callback", params={"code": "wrong-code", "state": state_of(link)}
    )
    assert refused.status_code == 502
    assert "Google did not accept the sign-in" in refused.text
    assert connector_states(client)["gmail"] == "not_configured"


def test_the_callback_shows_googles_error_as_plain_text(settings):
    client = chat_client(settings, [AIMessage(content="hi")])
    denied = client.get(
        "/auth/google/callback", params={"error": "<script>alert(1)</script>"}
    )
    assert denied.status_code == 400
    assert "<script>" not in denied.text
    assert "&lt;script&gt;" in denied.text


def test_the_callback_needs_no_api_key_but_everything_else_does(settings):
    settings.api_key = "secret"
    client = chat_client(settings, [AIMessage(content="hi")])
    assert client.post("/agent/chat", json={"message": "hi"}).status_code == 401
    # Google's redirect cannot carry the key; the one-time state guards it.
    reached = client.get("/auth/google/callback", params={"code": "x", "state": "y"})
    assert reached.status_code == 400


def test_an_email_asked_for_in_chat_is_sent_as_the_signed_in_account(settings):
    settings.actions_enabled = True
    settings.api_key = "secret"
    key = {"X-API-Key": "secret"}
    oauth = GoogleOAuth(settings)
    sign_in(oauth)

    sent = {}

    def gmail_api(request: httpx.Request) -> httpx.Response:
        sent["auth"] = request.headers["Authorization"]
        sent["url"] = str(request.url)
        return httpx.Response(200, json={"id": "msg1"})

    connectors = build_connectors(
        settings,
        clients={"gmail": httpx.Client(transport=httpx.MockTransport(gmail_api))},
        google_oauth=oauth,
    )
    script = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="request_send_email",
                    args={"to": "a@b.com", "subject": "Hi", "body": "Hello"},
                    id="t1",
                )
            ],
        ),
        AIMessage(content="I queued the email for your approval."),
    ]
    client = chat_client(settings, script, connectors=connectors, google_oauth=oauth)

    body = client.post("/agent/chat", json={"message": "email a@b.com"}, headers=key).json()
    assert sent == {}  # queued, not sent
    action_id = body["pending_action_ids"][0]

    decided = client.post(f"/actions/{action_id}/approve", headers=key).json()
    assert decided["status"] == "executed"
    assert sent["auth"] == "Bearer access-1"
    assert sent["url"].endswith("/messages/send")
