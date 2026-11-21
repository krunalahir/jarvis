from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.schemas import Claim, Platform, Source


@pytest.fixture
def client():
    store = Store(":memory:")
    provider = StaticProvider(
        default=ProviderResult(
            summary="s [1]",
            claims=[Claim(text="s", source_indexes=[0])],
            sources=[Source(title="t", url="https://example.com")],
        )
    )
    app = create_app(
        settings=Settings(actions_enabled=False),
        pipeline=ResearchPipeline(provider, store),
        store=store,
    )
    return TestClient(app)


def test_health(client):
    body = client.get("/health").json()
    assert body["ok"] is True
    assert body["actions_enabled"] is False


def test_ask_and_fetch_answer(client):
    response = client.post("/research/ask", json={"question": "test question"})
    assert response.status_code == 200
    answer = response.json()
    assert answer["status"] == "answered"
    fetched = client.get(f"/research/{answer['id']}")
    assert fetched.status_code == 200
    assert fetched.json()["question"] == "test question"
    assert client.get("/research").json()[0]["id"] == answer["id"]


def test_connector_status_listed(client):
    states = {c["name"]: c["state"] for c in client.get("/connectors/status").json()}
    assert states["gmail"] == "not_configured"


def test_draft_and_schedule_flow(client):
    draft = client.post(
        "/content/drafts", json={"platform": "x", "text": "hello"}
    ).json()
    run_at = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    scheduled = client.post(
        f"/content/drafts/{draft['id']}/schedule", json={"run_at": run_at}
    ).json()
    assert scheduled["status"] == "scheduled"

    # Not due yet: publish-due touches nothing
    assert client.post("/content/publish-due").json() == []

    # Backdate it: now it is due, and fails visibly (X not configured)
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    client.post(f"/content/drafts/{draft['id']}/schedule", json={"run_at": past})
    touched = client.post("/content/publish-due").json()
    assert touched[0]["status"] == "failed"
    assert "missing settings" in touched[0]["error"]


def test_api_key_enforced_when_set():
    store = Store(":memory:")
    app = create_app(
        settings=Settings(api_key="secret"),
        pipeline=ResearchPipeline(StaticProvider(default=ProviderResult()), store),
        store=store,
    )
    c = TestClient(app)
    assert c.post("/research/ask", json={"question": "q"}).status_code == 401
    ok = c.post("/research/ask", json={"question": "q"}, headers={"X-API-Key": "secret"})
    assert ok.status_code == 200


def test_actions_enabled_without_api_key_refuses_to_start():
    with pytest.raises(RuntimeError, match="JARVIS_API_KEY"):
        create_app(
            settings=Settings(actions_enabled=True),
            store=Store(":memory:"),
        )


def test_every_effect_is_pending_until_approved(client):
    # Requesting a send does NOT send: it creates a pending action.
    response = client.post(
        "/gmail/send",
        json={"to": "a@b.com", "subject": "Hi", "body": "hello"},
    )
    assert response.status_code == 202
    action = response.json()
    assert action["status"] == "pending"
    assert "a@b.com" in action["summary"]

    # Approving executes; here nobody has signed in to Google, so it fails
    # visibly and says what to do.
    decided = client.post(f"/actions/{action['id']}/approve").json()
    assert decided["status"] == "failed"
    assert "Google is not connected" in decided["error"]

    # A decided action cannot be re-approved or rejected.
    assert client.post(f"/actions/{action['id']}/approve").status_code == 409
    assert client.post(f"/actions/{action['id']}/reject").status_code == 409


def test_reject_blocks_execution(client):
    action = client.post(
        "/spotify/play", json={"track_query": "song"}
    ).json()
    decided = client.post(f"/actions/{action['id']}/reject").json()
    assert decided["status"] == "rejected"
    assert client.post(f"/actions/{action['id']}/approve").status_code == 409


def test_content_generate_unavailable_without_key():
    # Hosted OpenAI is opt-in; without its key, generation is a clean 409.
    app = create_app(
        settings=Settings(actions_enabled=False, llm_provider="openai"),
        store=Store(":memory:"),
    )
    response = TestClient(app).post(
        "/content/generate", json={"platform": "x", "topic": "launch day"}
    )
    assert response.status_code == 409
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_agent_chat_endpoint_runs_loop_and_reports_pending_actions():
    from langchain_core.messages import AIMessage, ToolCall

    from jarvis.agent import build_agent

    from fakes import ScriptedChatModel

    store = Store(":memory:")
    script = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="request_spotify_play",
                    args={"track_query": "hanuman chalisa"},
                    id="t1",
                )
            ],
        ),
        AIMessage(content="Queued for your approval; nothing is playing yet."),
    ]
    model = ScriptedChatModel(script=script)
    app = create_app(
        settings=Settings(actions_enabled=False),
        store=store,
        agent_builder=lambda tools: build_agent(Settings(), tools, model=model),
    )
    response = TestClient(app).post("/agent/chat", json={"message": "play hanuman chalisa"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["steps"][0]["tool"] == "request_spotify_play"
    assert len(body["pending_action_ids"]) == 1
    action = store.get_action(body["pending_action_ids"][0])
    assert action.status.value == "pending"  # still waiting for the user
