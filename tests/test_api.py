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
    assert client.get("/health").json() == {"ok": True, "actions_enabled": False}


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
