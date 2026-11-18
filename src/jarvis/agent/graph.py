"""The JARVIS agent loop, built with LangChain and LangGraph.

`create_agent` (LangChain 1.0) compiles a LangGraph state graph that runs
the loop the user asked for: the model plans, calls one tool, reads the
result, and decides the next step, until it answers or the bound stops
it. The model is the configured chat model - self-hosted Qwen3 8B by
default. Every write goes through the approval-gated request tools, so
the loop can act only after the user approves each specific action.
"""

from __future__ import annotations

import logging
from typing import Optional

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.errors import GraphRecursionError

from ..config import Settings
from ..llm import build_chat_model
from ..schemas import AgentReply, AgentRunStatus, AgentStep, Route, TaskStage

log = logging.getLogger(__name__)

# Prior turns kept per chat request; older ones are dropped.
MAX_HISTORY_MESSAGES = 20

SYSTEM_PROMPT = """You are JARVIS, a personal AI assistant.

How you work:
- Think, then act: use tools to gather facts before answering questions
  about current events, email, calendar or connected apps.
- Use web_search or research for anything recent or external. Never
  invent current facts, links or numbers; if search is not configured or
  finds nothing, say so plainly.
- Actions that change the world (send, post, publish, call, play) are
  only ever requested through the request_* tools. They queue a pending
  action for the user to approve. Never claim something was sent,
  posted, published, called or played; say it was queued for approval.
- Keep answers short and direct. Cite source URLs when you used search.
"""


def build_agent(
    settings: Settings,
    tools: list,
    model: Optional[BaseChatModel] = None,
    system_prompt: Optional[str] = None,
):
    """Compile the LangGraph agent graph for the configured model.

    `system_prompt` overrides the default; the orchestrator passes the
    base prompt plus the current route's instructions.
    """
    chat_model = model or build_chat_model(settings)
    return create_agent(
        model=chat_model, tools=tools, system_prompt=system_prompt or SYSTEM_PROMPT
    )


def run_agent(
    agent,
    message: str,
    max_steps: int = 8,
    history: Optional[list[dict]] = None,
    pending_action_ids: Optional[list] = None,
) -> AgentReply:
    """Run one bounded agent turn.

    The loop is bounded two ways: LangGraph's recursion limit caps the
    number of graph steps, and hitting it is reported as
    STOPPED_STEP_LIMIT instead of hanging or crashing.
    """
    message = message.strip()
    if not message:
        raise ValueError("message must not be empty")
    # Keep only the most recent turns: an unbounded history can overflow
    # the model's context window. Callers validate roles; this bounds size.
    history = list(history or [])[-MAX_HISTORY_MESSAGES:]
    messages = history + [{"role": "user", "content": message}]
    # Each plan/tool round is roughly two graph steps; +2 covers entry
    # and the final answer.
    recursion_limit = max(4, 2 * max_steps + 2)
    try:
        result = agent.invoke(
            {"messages": messages}, config={"recursion_limit": recursion_limit}
        )
    except GraphRecursionError:
        return AgentReply(
            reply=(
                f"I hit my safety limit of {max_steps} steps for one request, "
                "so I stopped instead of running indefinitely. Ask me to "
                "continue, or narrow the task."
            ),
            status=AgentRunStatus.STOPPED_STEP_LIMIT,
            steps=_trace(result_messages=None),
            pending_action_ids=list(pending_action_ids or []),
            detail="LangGraph recursion limit reached; the loop was stopped.",
        )
    except Exception as exc:
        log.warning("agent run failed: %s", exc)
        return AgentReply(
            reply=f"I could not complete that: {exc}",
            status=AgentRunStatus.FAILED,
            pending_action_ids=list(pending_action_ids or []),
            detail=str(exc),
        )

    out_messages = result.get("messages", [])
    reply_text = ""
    for msg in reversed(out_messages):
        if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
            reply_text = msg.content if isinstance(msg.content, str) else str(msg.content)
            break
    return AgentReply(
        reply=reply_text or "(the model returned no text answer)",
        status=AgentRunStatus.ANSWERED,
        steps=_trace(out_messages),
        pending_action_ids=list(pending_action_ids or []),
    )


def _trace(result_messages: Optional[list]) -> list[AgentStep]:
    steps = []
    for msg in result_messages or []:
        if isinstance(msg, AIMessage):
            for call in msg.tool_calls or []:
                steps.append(
                    AgentStep(tool=call.get("name", ""), arguments=call.get("args", {}))
                )
    return steps




# Route-specific instructions appended to the base system prompt, so each
# task's agent stays focused on the job the router gave it.
ROUTE_PROMPTS: dict[Route, str] = {
    Route.RESEARCH: (
        "\nThis request is a research task. Use the research tool for the "
        "deep cited answer; use web_search only for a quick check. Base the "
        "reply on what the tools return and cite source URLs."
    ),
    Route.WEB_SEARCH: (
        "\nThis request is a quick current-facts lookup. Use web_search once, "
        "answer from the results, and cite source URLs."
    ),
    Route.GMAIL: (
        "\nThis request is an email task. Use read_emails to look things up. "
        "Sending is only ever queued through request_send_email for the user "
        "to approve. If Gmail is not configured, say exactly what is missing."
    ),
    Route.CALENDAR: (
        "\nThis request is a calendar task. Use list_calendar_events to look "
        "things up. Creating events is only ever queued through "
        "request_create_event for the user to approve."
    ),
    Route.PHONE_CALL: (
        "\nThis request is a phone-call task. Calls are only ever queued "
        "through request_phone_call for the user to approve; confirm what "
        "will be said in the request."
    ),
    Route.SPOTIFY: (
        "\nThis request is a Spotify task. Playback is only ever queued "
        "through request_spotify_play for the user to approve."
    ),
    Route.SOCIAL_CONTENT: (
        "\nThis request is a social-content task. Use draft_post to write "
        "drafts (drafts never publish themselves). Publishing is only ever "
        "queued through request_publish_draft for the user to approve."
    ),
    Route.APP_STATUS: (
        "\nThis request asks what is connected. Use connector_status and "
        "report each app's state plainly."
    ),
    Route.DIRECT_ANSWER: (
        "\nThis request needs no tool. Answer directly from your own "
        "knowledge, and say so when something would need a live lookup."
    ),
}


def run_task(
    settings: Settings,
    message: str,
    history: Optional[list[dict]] = None,
    max_steps: Optional[int] = None,
    *,
    store=None,
    pipeline=None,
    gate=None,
    scheduler=None,
    generator=None,
    connectors=None,
    tools_builder=None,
    agent_builder=None,
    router=None,
    monitor=None,
    model: Optional[BaseChatModel] = None,
) -> AgentReply:
    """Run one chat message through the full JARVIS workflow.

    The shape every task follows:

    1. The monitor records the request and opens a task.
    2. The router reads the message and picks exactly one route.
    3. For app routes the monitor checks authentication up front and
       tells the user what is missing (connections run on their saved
       credentials, so no sign-in step happens mid-task).
    4. A bounded agent loop executes the task with only that route's
       tools; a direct answer runs with no tools at all.
    5. The monitor records each tool step and every approval the agent
       queues, retries a failed run once (recovery), and closes the task
       with the outcome.

    The reply carries the route, the task id and the monitor's updates,
    so the chat client can show the whole trail.
    """
    from .monitor import TaskMonitor
    from .router import Router
    from .tools import build_tools as _build_tools

    message = message.strip()
    if not message:
        raise ValueError("message must not be empty")
    max_steps = max_steps if max_steps is not None else settings.agent_max_steps
    monitor = monitor or TaskMonitor(store=store)
    router = router or Router(settings, model=model)

    monitor.update(TaskStage.RECEIVED, f"Working on: {message[:120]}")
    decision = router.classify(message)
    monitor.update(
        TaskStage.ROUTED,
        f"Route: {decision.route.value} - {decision.reason or 'picked by the router'}",
    )

    pending: list = []
    tools = []
    if decision.route != Route.DIRECT_ANSWER:
        # Auth supervision for app routes: report readiness before work
        # starts. The agent still runs, because the tools themselves
        # explain a missing connection and read-only steps may not need
        # actions to be enabled.
        if decision.route == Route.GMAIL and connectors is not None:
            monitor.check_auth(connectors["gmail"])
        elif decision.route == Route.CALENDAR and connectors is not None:
            monitor.check_auth(connectors["google_calendar"])
        elif decision.route == Route.PHONE_CALL and connectors is not None:
            monitor.check_auth(connectors["calling"])
        elif decision.route == Route.SPOTIFY and connectors is not None:
            monitor.check_auth(connectors["spotify"])
        builder = tools_builder or _build_tools
        tools = builder(
            settings,
            pipeline,
            gate,
            scheduler,
            generator,
            connectors,
            pending_action_ids=pending,
            route=decision.route,
        )
    prompt = SYSTEM_PROMPT + ROUTE_PROMPTS[decision.route]

    def _attempt():
        if agent_builder is not None:
            agent = agent_builder(tools)
        else:
            agent = build_agent(settings, tools, model=model, system_prompt=prompt)
        return run_agent(
            agent,
            message,
            max_steps=max_steps,
            history=history,
            pending_action_ids=pending,
        )

    monitor.update(TaskStage.EXECUTING, "Task started.")
    reply = _attempt()
    attempt = 0
    while monitor.should_retry(reply, attempt):
        attempt += 1
        monitor.update(
            TaskStage.RECOVERING,
            f"Something failed ({reply.detail or reply.status.value}); trying once more.",
        )
        reply = _attempt()

    for step in reply.steps:
        monitor.update(TaskStage.STEP, f"Used tool: {step.tool}")
    for action_id in reply.pending_action_ids:
        monitor.update(
            TaskStage.APPROVAL_NEEDED,
            f"Waiting on your approval: pending action {action_id}. "
            "Nothing happens until you approve it.",
        )
    if reply.status == AgentRunStatus.FAILED:
        monitor.update(TaskStage.FAILED, f"Task failed: {reply.detail or 'unknown error'}")
    else:
        monitor.update(TaskStage.COMPLETED, "Task complete.")

    reply.route = decision.route
    reply.task_id = monitor.task_id
    reply.updates = monitor.updates
    return reply
