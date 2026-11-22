"""Conversation memory: summary + recent turns + exact facts, and the chat page."""

from typing import Any

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, ToolCall

from jarvis.actions import ActionGate
from jarvis.agent import build_agent, build_tools, run_agent
from jarvis.api import create_app
from jarvis.config import Settings
from jarvis.connectors import build_connectors
from jarvis.content import ContentGenerator
from jarvis.memory import MemoryManager, extract_facts
from jarvis.research import ResearchPipeline, StaticProvider, Store
from jarvis.research.providers import ProviderResult
from jarvis.scheduler import PublishScheduler
from jarvis.schemas import ActionKind, ConversationMemory

from fakes import ScriptedChatModel


class RecordingModel(ScriptedChatModel):
    """A scripted model that also keeps the text of everything it was sent."""

    seen: Any = None

    def _generate(self, messages, stop=None, **kwargs):
        self.seen.append(
            "\n".join(
                str(m[1] if isinstance(m, tuple) else getattr(m, "content", m))
                for m in messages
            )
        )
        return super()._generate(messages, stop=stop, **kwargs)


def summariser(*replies):
    return RecordingModel(script=[AIMessage(content=r) for r in replies], seen=[])


# --- facts ---


def test_facts_are_copied_exactly():
    facts = extract_facts(
        "My name is Shivansh. Send it to krunal.ahir+test@gmail.com on 5th December 2026. "
        "His number is +91 98765 43210."
    )
    assert "email: krunal.ahir+test@gmail.com" in facts
    assert "date mentioned: 5th December 2026" in facts
    assert "phone: +91 98765 43210" in facts
    assert 'user said: "My name is Shivansh."' in facts


def test_plain_message_has_no_facts():
    assert extract_facts("what is the capital of Gujarat?") == []


# --- the three parts ---


def test_context_puts_facts_and_summary_before_the_turns():
    manager = MemoryManager(Store(":memory:"), Settings())
    memory = ConversationMemory(
        id="c1",
        summary="User planned a trip.",
        facts=["email: a@b.com"],
        turns=[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    )
    messages = manager.context_messages(memory)
    assert messages[0]["role"] == "system"
    assert "email: a@b.com" in messages[0]["content"]
    assert "User planned a trip." in messages[0]["content"]
    assert messages[1:] == memory.turns


def test_new_conversation_has_no_context():
    manager = MemoryManager(Store(":memory:"), Settings())
    assert manager.context_messages(manager.load("fresh")) == []


def test_bad_conversation_id_is_refused():
    manager = MemoryManager(Store(":memory:"), Settings())
    with pytest.raises(ValueError):
        manager.load("../../etc/passwd")


def test_turns_stay_word_for_word_until_the_limit():
    model = summariser("should not be used")
    store = Store(":memory:")
    manager = MemoryManager(store, Settings(), model=model)
    memory = manager.load("c1")
    for i in range(7):  # 14 messages: still within 15
        manager.record(memory, f"question {i}", f"answer {i}")
    assert model.calls == 0
    saved = store.get_conversation("c1")
    assert len(saved.turns) == 14
    assert saved.summary == ""
    assert saved.turns[0] == {"role": "user", "content": "question 0"}


def test_past_the_limit_old_turns_become_the_summary():
    model = summariser("They discussed questions 0 to 6.")
    store = Store(":memory:")
    manager = MemoryManager(store, Settings(), model=model)
    memory = manager.load("c1")
    for i in range(8):  # 16 messages: one past the limit
        manager.record(memory, f"question {i} for a@b.com", f"answer {i}")
    saved = store.get_conversation("c1")
    assert model.calls == 1  # summarised once, not on every message
    assert saved.summary == "They discussed questions 0 to 6."
    # the newest stay word for word, starting on a user message
    assert saved.turns == [
        {"role": "user", "content": "question 6 for a@b.com"},
        {"role": "assistant", "content": "answer 6"},
        {"role": "user", "content": "question 7 for a@b.com"},
        {"role": "assistant", "content": "answer 7"},
    ]
    assert "question 0" in model.seen[0]  # the old turns were given to the summariser
    assert saved.facts == ["email: a@b.com"]  # stated 8 times, stored once


def test_summary_is_carried_into_the_next_summary():
    model = summariser("first summary", "second summary")
    manager = MemoryManager(Store(":memory:"), Settings(), model=model)
    memory = manager.load("c1")
    for i in range(20):
        manager.record(memory, f"q{i}", f"a{i}")
    assert model.calls >= 2
    assert "first summary" in model.seen[1]
    assert memory.summary == "second summary"


def test_thinking_text_is_removed_from_the_summary():
    model = summariser("<think>hmm</think>Clean summary.")
    manager = MemoryManager(Store(":memory:"), Settings(), model=model)
    memory = manager.load("c1")
    for i in range(8):
        manager.record(memory, f"q{i}", f"a{i}")
    assert memory.summary == "Clean summary."


def test_if_the_summariser_fails_nothing_is_lost():
    model = RecordingModel(script=[RuntimeError("model is down")], seen=[])
    store = Store(":memory:")
    manager = MemoryManager(store, Settings(), model=model)
    memory = manager.load("c1")
    for i in range(10):
        manager.record(memory, f"q{i}", f"a{i}")
    saved = store.get_conversation("c1")
    assert saved.summary == ""
    assert len(saved.turns) == 20  # all kept, to be summarised next time
    for i in range(30):
        manager.record(memory, f"more{i}", f"a{i}")
    assert len(memory.turns) <= 45  # but never without end


def test_limits_come_from_settings():
    model = summariser("s")
    manager = MemoryManager(
        Store(":memory:"), Settings(memory_max_recent=6, memory_min_recent=2), model=model
    )
    memory = manager.load("c1")
    for i in range(4):
        manager.record(memory, f"q{i}", f"a{i}")
    assert model.calls == 1
    assert memory.turns == [
        {"role": "user", "content": "q3"},
        {"role": "assistant", "content": "a3"},
    ]


# --- through the API ---


def make_app(agent_model, memory_model=None, settings=None):
    store = Store(":memory:")
    settings = settings or Settings()
    app = create_app(
        settings=settings,
        store=store,
        agent_builder=lambda tools: build_agent(Settings(), tools, model=agent_model),
        memory=MemoryManager(store, settings, model=memory_model or summariser("s")),
    )
    return TestClient(app), store


def test_second_message_sees_the_first_without_sending_history():
    agent_model = RecordingModel(
        script=[AIMessage(content="Noted."), AIMessage(content="It is a@b.com.")], seen=[]
    )
    client, store = make_app(agent_model)
    first = client.post(
        "/agent/chat",
        json={"message": "My friend's mail is a@b.com", "conversation_id": "chat-1"},
    ).json()
    assert first["conversation_id"] == "chat-1"
    client.post(
        "/agent/chat", json={"message": "what was the mail?", "conversation_id": "chat-1"}
    )
    second_prompt = agent_model.seen[1]
    assert "My friend's mail is a@b.com" in second_prompt  # the turn, word for word
    assert "email: a@b.com" in second_prompt  # and the stored fact
    assert "Noted." in second_prompt

    memory = client.get("/conversations/chat-1").json()
    assert len(memory["turns"]) == 4
    assert memory["facts"] == ["email: a@b.com"]


def test_conversations_do_not_mix():
    agent_model = RecordingModel(script=[AIMessage(content="ok")], seen=[])
    client, _ = make_app(agent_model)
    client.post("/agent/chat", json={"message": "secret one", "conversation_id": "a"})
    client.post("/agent/chat", json={"message": "hello", "conversation_id": "b"})
    assert "secret one" not in agent_model.seen[1]


def test_without_conversation_id_nothing_is_remembered():
    agent_model = RecordingModel(script=[AIMessage(content="ok")], seen=[])
    client, store = make_app(agent_model)
    body = client.post(
        "/agent/chat",
        json={"message": "now", "history": [{"role": "user", "content": "earlier"}]},
    ).json()
    assert body["conversation_id"] is None
    assert "earlier" in agent_model.seen[0]  # old history style still works
    assert client.get("/conversations/anything").status_code == 404


def test_bad_conversation_id_gives_400():
    client, _ = make_app(RecordingModel(script=[AIMessage(content="ok")], seen=[]))
    response = client.post(
        "/agent/chat", json={"message": "hi", "conversation_id": "has spaces!"}
    )
    assert response.status_code == 400


def test_failed_run_is_not_remembered():
    client, store = make_app(RecordingModel(script=[RuntimeError("down")], seen=[]))
    body = client.post(
        "/agent/chat", json={"message": "hi", "conversation_id": "c1"}
    ).json()
    assert body["status"] == "failed"
    assert store.get_conversation("c1") is None


def test_queued_action_is_noted_in_memory():
    script = [
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="request_spotify_play", args={"track_query": "x"}, id="t1")],
        ),
        AIMessage(content="Queued."),
    ]
    client, store = make_app(RecordingModel(script=script, seen=[]))
    body = client.post(
        "/agent/chat", json={"message": "play x", "conversation_id": "c1"}
    ).json()
    last = store.get_conversation("c1").turns[-1]["content"]
    assert body["pending_action_ids"][0] in last
    assert "not executed yet" in last


def test_chat_page_is_served_and_talks_to_agent_chat():
    client, _ = make_app(RecordingModel(script=[AIMessage(content="ok")], seen=[]))
    response = client.get("/")
    assert response.status_code == 200
    assert "/agent/chat" in response.text
    assert "conversation_id" in response.text


def test_memory_route_needs_the_api_key():
    store = Store(":memory:")
    settings = Settings(api_key="k")
    app = create_app(settings=settings, store=store, memory=MemoryManager(store, settings))
    client = TestClient(app)
    assert client.get("/conversations/c1").status_code == 401
    assert client.get("/").status_code == 200  # the page itself is open


# --- testing switches in the agent tools ---


def tool_run(settings, script):
    store = Store(":memory:")
    connectors = build_connectors(settings)
    executed = []
    gate = ActionGate(store, {kind: (lambda p: executed.append(p) or {}) for kind in ActionKind})
    pending = []
    tools = build_tools(
        settings,
        ResearchPipeline(StaticProvider(default=ProviderResult(summary="s")), store),
        gate,
        PublishScheduler(store, connectors),
        ContentGenerator(settings, model=ScriptedChatModel(script=[AIMessage(content="d")])),
        connectors,
        pending_action_ids=pending,
    )
    agent = build_agent(settings, tools, model=ScriptedChatModel(script=script))
    return agent, store, pending, executed


def play_script():
    return [
        AIMessage(
            content="",
            tool_calls=[ToolCall(name="request_spotify_play", args={"track_query": "x"}, id="t1")],
        ),
        AIMessage(content="done"),
    ]


def test_auto_approve_executes_at_once():
    agent, store, pending, executed = tool_run(Settings(auto_approve=True), play_script())
    run_agent(agent, "play x", pending_action_ids=pending)
    assert executed == [{"track_query": "x"}]
    assert store.get_action(pending[0]).status.value == "executed"


def test_without_auto_approve_nothing_executes():
    agent, store, pending, executed = tool_run(Settings(), play_script())
    run_agent(agent, "play x", pending_action_ids=pending)
    assert executed == []
    assert store.get_action(pending[0]).status.value == "pending"


def test_email_with_a_placeholder_is_not_queued():
    script = [
        AIMessage(
            content="",
            tool_calls=[
                ToolCall(
                    name="request_send_email",
                    args={"to": "a@b.com", "subject": "Hi", "body": "Dear [Friend's Name], ..."},
                    id="t1",
                )
            ],
        ),
        AIMessage(content="What is your friend's name?"),
    ]
    agent, store, pending, executed = tool_run(Settings(auto_approve=True), script)
    run_agent(agent, "mail my friend", pending_action_ids=pending)
    assert pending == [] and executed == []
    assert store.list_actions() == []
