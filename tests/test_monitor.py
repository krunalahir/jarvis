"""Task monitor tests: progress trail, auth supervision, recovery policy.

All offline: the store is in-memory and the connector is unconfigured.
"""

from jarvis.actions import ActionGate
from jarvis.agent.monitor import TaskMonitor
from jarvis.config import Settings
from jarvis.connectors import build_connectors
from jarvis.research import Store
from jarvis.schemas import AgentReply, AgentRunStatus, TaskStage


def make_monitor():
    store = Store(":memory:")
    return TaskMonitor(store=store), store


def test_updates_are_sequenced_and_chat_ready():
    monitor, _ = make_monitor()
    monitor.update(TaskStage.RECEIVED, "Got your message.")
    monitor.update(TaskStage.ROUTED, "Route: gmail.")
    updates = monitor.updates
    assert [u.seq for u in updates] == [1, 2]
    assert updates[0].stage == TaskStage.RECEIVED
    assert updates[1].message == "Route: gmail."
    assert all(u.task_id == monitor.task_id for u in updates)


def test_updates_persist_to_the_store():
    monitor, store = make_monitor()
    monitor.update(TaskStage.RECEIVED, "Got your message.")
    monitor.update(TaskStage.COMPLETED, "Done.")
    stored = store.list_task_updates(monitor.task_id)
    assert [u.stage for u in stored] == [TaskStage.RECEIVED, TaskStage.COMPLETED]
    assert stored[1].message == "Done."
    # other tasks do not leak in
    assert store.list_task_updates("someone-else") == []


def test_auth_check_reports_missing_credentials():
    monitor, _ = make_monitor()
    connectors = build_connectors(Settings())
    ready = monitor.check_auth(connectors["gmail"])
    assert ready is False
    update = monitor.updates[-1]
    assert update.stage == TaskStage.AUTH_CHECK
    assert "GOOGLE_ACCESS_TOKEN" in update.message
    assert "automatically" in update.message


def test_auth_check_passes_when_ready():
    monitor, _ = make_monitor()
    settings = Settings(google_access_token="tok", actions_enabled=True)
    connectors = build_connectors(settings)
    assert monitor.check_auth(connectors["gmail"]) is True
    assert "ready" in monitor.updates[-1].message


def test_auth_check_notes_actions_disabled():
    monitor, _ = make_monitor()
    settings = Settings(google_access_token="tok", actions_enabled=False)
    connectors = build_connectors(settings)
    assert monitor.check_auth(connectors["gmail"]) is False
    assert "disabled" in monitor.updates[-1].message


def test_failed_run_retried_once_then_stops():
    monitor, _ = make_monitor()
    failed = AgentReply(reply="boom", status=AgentRunStatus.FAILED, detail="x")
    assert monitor.should_retry(failed, attempt=0) is True
    assert monitor.should_retry(failed, attempt=1) is False


def test_step_limit_stop_is_not_retried():
    monitor, _ = make_monitor()
    stopped = AgentReply(reply="stopped", status=AgentRunStatus.STOPPED_STEP_LIMIT)
    assert monitor.should_retry(stopped, attempt=0) is False


def test_monitor_works_without_a_store():
    monitor = TaskMonitor()
    monitor.update(TaskStage.RECEIVED, "hi")
    assert len(monitor.updates) == 1
