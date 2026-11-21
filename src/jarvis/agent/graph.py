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
from ..schemas import AgentReply, AgentRunStatus, AgentStep

log = logging.getLogger(__name__)

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
- Google apps (Gmail, Calendar, YouTube) work only after the user signs
  in. When the user asks to sign in or connect Google, or a tool says
  Google is not connected, call connect_google and give the user the
  link it returns, copied exactly. Only the user can open the link and
  press Allow; never say Google is connected unless a tool said so.
- When you queue an email, show the user the subject and body you wrote
  so they can check it before approving.
- Keep answers short and direct. Cite source URLs when you used search.
"""


def build_agent(settings: Settings, tools: list, model: Optional[BaseChatModel] = None):
    """Compile the LangGraph agent graph for the configured model."""
    chat_model = model or build_chat_model(settings)
    return create_agent(model=chat_model, tools=tools, system_prompt=SYSTEM_PROMPT)


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
    messages = list(history or []) + [{"role": "user", "content": message}]
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
