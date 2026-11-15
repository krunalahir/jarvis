"""Agent loop tests: iteration, termination, and write-action safety.

All offline: the model is a scripted fake and no connector is configured.
"""

import pytest
from langchain_core.messages import AIMessage, ToolCall

from jarvis.actions import ActionGate
from jarvis.agent import build_agent, build_tools, run_agent
from jarvis.config import Settings
from jarvis.connectors import build_connectors
from jarvis.content import ContentGenerator
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.scheduler import PublishScheduler
from jarvis.schemas import ActionKind, AgentRunStatus, Claim, PendingActionStatus, Source

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
    generator = ContentGenerator(settings, model=ScriptedChatModel(script=[AIMessage(content="draft text")]))
    executed = []
    gate = ActionGate(
        store,
        {kind: (lambda p: executed.append(p) or {}) for kind in ActionKind},
    )
    yield settings, store, pipeline, connectors, scheduler, generator, gate, executed
    store.close()


def make_run(settings, store, pipeline, connectors, scheduler, generator, gate, script):
    pending = []
    tools = build_tools(
        settings, pipeline, gate, scheduler, generator, connectors, pending_action_ids=pending
    )
    agent = build_agent(settings, tools, model=ScriptedChatModel(script=script))
    return agent, pending


def test_agent_iterates_tool_then_answers(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    script = [
        AIMessage(content="", tool_calls=[ToolCall(name="web_search", args={"query": "news"}, id="t1")]),
        AIMessage(content="Web search is not configured, so I cannot check fresh news yet."),
    ]
    agent, pending = make_run(settings, store, pipeline, connectors, scheduler, generator, gate, script)
    reply = run_agent(agent, "what is in the news?", max_steps=6, pending_action_ids=pending)
    assert reply.status == AgentRunStatus.ANSWERED
    assert reply.steps[0].tool == "web_search"
    assert "not configured" in reply.reply


def test_write_tool_queues_pending_action_and_never_executes(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, executed = parts
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
        AIMessage(content="I queued the email for your approval; nothing was sent."),
    ]
    agent, pending = make_run(settings, store, pipeline, connectors, scheduler, generator, gate, script)
    reply = run_agent(agent, "email a@b.com", max_steps=6, pending_action_ids=pending)
    assert reply.status == AgentRunStatus.ANSWERED
    assert len(reply.pending_action_ids) == 1
    action = store.get_action(reply.pending_action_ids[0])
    assert action.status == PendingActionStatus.PENDING  # waiting for the user
    assert executed == []  # the executor never ran


def test_runaway_loop_is_stopped_by_the_bound(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    script = [
        AIMessage(content="", tool_calls=[ToolCall(name="connector_status", args={}, id="t1")]),
    ]
    agent, pending = make_run(settings, store, pipeline, connectors, scheduler, generator, gate, script)
    reply = run_agent(agent, "loop forever", max_steps=3, pending_action_ids=pending)
    assert reply.status == AgentRunStatus.STOPPED_STEP_LIMIT
    assert "safety limit" in reply.reply


def test_research_tool_returns_stored_cited_answer(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    script = [
        AIMessage(content="", tool_calls=[ToolCall(name="research", args={"question": "largest city in Gujarat?"}, id="t1")]),
        AIMessage(content="Ahmedabad, per the research answer [1]."),
    ]
    agent, pending = make_run(settings, store, pipeline, connectors, scheduler, generator, gate, script)
    reply = run_agent(agent, "largest city in Gujarat?", max_steps=6, pending_action_ids=pending)
    assert reply.status == AgentRunStatus.ANSWERED
    assert reply.steps[0].tool == "research"
    answers = store.list_answers()
    assert answers and answers[0].sources[0].url == "https://example.com/amd"


def test_empty_message_rejected(parts):
    settings, store, pipeline, connectors, scheduler, generator, gate, _ = parts
    agent, pending = make_run(settings, store, pipeline, connectors, scheduler, generator, gate, [AIMessage(content="x")])
    with pytest.raises(ValueError):
        run_agent(agent, "   ", pending_action_ids=pending)
