"""Orchestration tests: router -> scoped agent -> monitor, end to end.

All offline: router and agent models are scripted fakes; the store is
in-memory; no connector is configured.
"""

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolCall

from jarvis.actions import ActionGate
from jarvis.agent import Router, build_agent, build_tools, run_task
from jarvis.agent.tools import ROUTE_TOOLS
from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.connectors import build_connectors
from jarvis.content import ContentGenerator
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.scheduler import PublishScheduler
from jarvis.schemas import (
    ActionKind,
    AgentRunStatus,
    Claim,
    PendingActionStatus,
    Route,
    Source,
    TaskStage,
)

from fakes import ScriptedChatModel


@pytest.fixture
def parts():
    settings = Settings()
    store = Store(":memory:")
    pipeline = ResearchPipeline(
        StaticProvider(
            default=ProviderResult(
                summary="Ahmedabad is in Gujarat. [1]",
                claims=[Claim(text="Ahmedabad is in Gujarat.", source_indexes=[0])],
                sources=[Source(title="Ahmedabad", url="https://example.com/amd")],
            )
        ),
        store,
    )
    connectors = build_connectors(settings)
    scheduler = PublishScheduler(store, connectors)
    generator = ContentGenerator(settings, model=ScriptedChatModel(script=[AIMessage(content="draft")]))
    executed = []
    gate = ActionGate(
        store,
        {kind: (lambda p: executed.append(p) or {}) for kind in ActionKind},
    )
    yield settings, store, pipeline, connectors, scheduler, generator, gate, executed
    store.close()


def make_run(parts, router_script, agent_script):
    settings, store, pipeline, connectors, scheduler, generator, gate, executed = parts
    router = Router(settings, model=ScriptedChatModel(script=router_script))
    agent_model = ScriptedChatModel(script=agent_script)
    seen_tools = {}

    def agent_builder(tools):
        seen_tools["names"] = sorted(t.name for t in tools)
        return build_agent(settings, tools, model=agent_model)

    def go(message):
        return run_task(
            settings,
            message,
            store=store,
            pipeline=pipeline,
            gate=gate,
            scheduler=scheduler,
            generator=generator,
            connectors=connectors,
            agent_builder=agent_builder,
            router=router,
        )

    return go, seen_tools, store, executed


def routing(text):
    return [AIMessage(content=text)]


def test_gmail_request_routes_to_gmail_tools_only(parts):
    go, seen_tools, store, _ = make_run(
        parts,
        routing('{"route": "gmail", "reason": "inbox question"}'),
        [
            AIMessage(content="", tool_calls=[ToolCall(name="read_emails", args={"query": "priya"}, id="t1")]),
            AIMessage(content="Gmail is not configured yet, so I cannot read the inbox."),
        ],
    )
    reply = go("any mail from priya?")
    assert reply.route == Route.GMAIL
    assert seen_tools["names"] == sorted(ROUTE_TOOLS[Route.GMAIL])
    assert "not configured" in reply.reply


def test_direct_answer_uses_no_tools(parts):
    go, seen_tools, _, _ = make_run(
        parts,
        routing('{"route": "direct_answer", "reason": "pure conversation"}'),
        [AIMessage(content="I think remote work depends on the team.")],
    )
    reply = go("what do you think about remote work?")
    assert reply.route == Route.DIRECT_ANSWER
    assert seen_tools["names"] == []
    assert reply.steps == []
    assert "remote work" in reply.reply


def test_research_request_gets_research_toolset(parts):
    go, seen_tools, _, _ = make_run(
        parts,
        routing('{"route": "research", "reason": "needs depth"}'),
        [
            AIMessage(content="", tool_calls=[ToolCall(name="research", args={"question": "q"}, id="t1")]),
            AIMessage(content="Here is the cited answer."),
        ],
    )
    reply = go("explain inflation in depth")
    assert reply.route == Route.RESEARCH
    assert seen_tools["names"] == sorted(ROUTE_TOOLS[Route.RESEARCH])


def test_monitor_trail_covers_the_whole_task(parts):
    go, _, store, _ = make_run(
        parts,
        routing('{"route": "gmail", "reason": "send mail"}'),
        [
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
            AIMessage(content="Queued the email for your approval."),
        ],
    )
    reply = go("email a@b.com saying hi")
    stages = [u.stage for u in reply.updates]
    assert stages[0] == TaskStage.RECEIVED
    assert TaskStage.ROUTED in stages
    assert TaskStage.AUTH_CHECK in stages  # gmail route checks connection first
    assert TaskStage.EXECUTING in stages
    assert TaskStage.STEP in stages
    assert TaskStage.APPROVAL_NEEDED in stages
    assert stages[-1] == TaskStage.COMPLETED
    assert reply.task_id
    # the trail is persisted for polling after the fact
    stored = store.list_task_updates(reply.task_id)
    assert [u.seq for u in stored] == [u.seq for u in reply.updates]
    # the write still only queued, never executed
    action = store.get_action(reply.pending_action_ids[0])
    assert action.status == PendingActionStatus.PENDING


def test_auth_update_names_missing_credential(parts):
    go, _, _, _ = make_run(
        parts,
        routing('{"route": "gmail", "reason": "inbox"}'),
        [AIMessage(content="Gmail is not configured.")],
    )
    reply = go("read my inbox")
    auth = [u for u in reply.updates if u.stage == TaskStage.AUTH_CHECK]
    assert auth and "GOOGLE_ACCESS_TOKEN" in auth[0].message


def test_failed_run_recovers_once_then_reports(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    router = Router(settings, model=ScriptedChatModel(script=routing('{"route": "direct_answer", "reason": "chat"}')))
    calls = []

    class FlakyAgent:
        def invoke(self, *a, **k):
            calls.append(1)
            raise RuntimeError("model server hiccup")

    reply = run_task(
        settings,
        "hello",
        store=store,
        pipeline=pipeline,
        gate=gate,
        scheduler=scheduler,
        generator=generator,
        connectors=connectors,
        agent_builder=lambda tools: FlakyAgent(),
        router=router,
    )
    assert reply.status == AgentRunStatus.FAILED
    assert len(calls) == 2  # one initial attempt + one recovery retry
    stages = [u.stage for u in reply.updates]
    assert TaskStage.RECOVERING in stages
    assert stages[-1] == TaskStage.FAILED


def test_chat_api_returns_route_and_updates(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    model = ScriptedChatModel(
        script=[
            AIMessage(content="", tool_calls=[ToolCall(name="request_spotify_play", args={"track_query": "lofi"}, id="t1")]),
            AIMessage(content="Queued Spotify playback for your approval."),
        ]
    )
    app = create_app(
        settings=settings,
        store=store,
        agent_builder=lambda tools: build_agent(Settings(), tools, model=model),
    )
    client = TestClient(app)
    response = client.post("/agent/chat", json={"message": "play some lofi"})
    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "spotify"
    assert body["task_id"]
    assert body["status"] == "answered"
    stages = [u["stage"] for u in body["updates"]]
    assert stages[0] == "received" and stages[-1] == "completed"
    assert "approval_needed" in stages
    # polling endpoint returns the same persisted trail
    polled = client.get(f"/agent/tasks/{body['task_id']}/updates")
    assert polled.status_code == 200
    assert [u["seq"] for u in polled.json()] == [u["seq"] for u in body["updates"]]
    assert client.get("/agent/tasks/nope/updates").status_code == 404


def test_empty_message_rejected(parts):
    go, _, _, _ = make_run(parts, routing('{"route": "direct_answer"}'), [AIMessage(content="x")])
    with pytest.raises(ValueError):
        go("   ")
