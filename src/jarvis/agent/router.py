"""The request router: one chat box, and JARVIS picks the right tool.

Every chat message meets the router first. It reads the request and
picks exactly one route: research, a quick web search, Gmail, calendar,
a phone call, Spotify, social content, an app-status check, or a direct
answer when no tool fits. Execution then happens in the agent loop with
only that route's tools (agent/graph.py), and the task monitor watches
the run end to end (agent/monitor.py).

Routing is a pure read: no tool runs and nothing executes here. The
configured chat model does the classification (self-hosted Qwen3 8B by
default; any OpenAI-compatible endpoint works, so the free hosted Gemini
testing path drives it unchanged). If the model's answer cannot be
parsed, or the model is unreachable, a deterministic keyword fallback
classifies instead, so routing never hard-fails. When nothing matches,
the message goes to a direct answer with no tools.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..llm import build_chat_model
from ..schemas import Route, RouteDecision

log = logging.getLogger(__name__)

ROUTING_PROMPT = """You are the request router for JARVIS, a personal AI assistant.

Read the user's message and pick exactly ONE route for it:

- research: a question that needs depth, explanation or cited sources
  ("explain", "compare", "how does X work", "research Y").
- web_search: a quick lookup of something current or external
  (news, weather, prices, "latest", "today", "who won").
- gmail: anything about email - reading the inbox or sending mail.
- calendar: events, meetings, schedule, "what's on my calendar".
- phone_call: making a phone call.
- spotify: playing music.
- social_content: writing, scheduling or publishing posts for
  Instagram, X, YouTube or Snapchat.
- app_status: which apps are connected or ready ("what can you do",
  "is Gmail connected").
- direct_answer: everything else - greetings, opinions, math, advice,
  or anything the assistant can answer from its own knowledge with no
  tool at all.

Rules:
- Pick exactly one route. When in doubt between a tool route and
  direct_answer, prefer direct_answer for pure conversation.
- Answer with ONLY a JSON object on one line, no other text:
  {"route": "<route>", "reason": "<one short sentence>"}
"""

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)

# Keyword fallback, checked in order: the first matching group wins.
# Social content comes before calendar so "schedule a post" does not
# become a calendar event; Gmail before research so "research this email"
# still lands in the mailbox.
_HEURISTIC_ROUTES: tuple[tuple[Route, tuple[str, ...]], ...] = (
    (Route.SOCIAL_CONTENT, ("instagram", "snapchat", "tweet", "hashtag",
                            "post about", "write a post", "draft a post",
                            "youtube video", "publish", "social media")),
    (Route.GMAIL, ("gmail", "email", "e-mail", "inbox", "mail from",
                   "unread")),
    (Route.CALENDAR, ("calendar", "meeting", "event", "schedule a call",
                      "my schedule", "appoint")),
    (Route.PHONE_CALL, ("phone call", "call ", "dial", "ring ")),
    (Route.SPOTIFY, ("spotify", "play music", "play a song", "play the song",
                     "play some", "song by")),
    (Route.APP_STATUS, ("what's connected", "what is connected", "connected apps",
                        "app status", "connector status", "what can you do",
                        "which apps")),
    (Route.RESEARCH, ("research", "explain", "compare", "in depth",
                      "detailed answer", "sources", "investigate",
                      "how does", "why does", "what causes")),
    (Route.WEB_SEARCH, ("news", "latest", "today", "current", "weather",
                        "price of", "stock", "who won", "score", "search for",
                        "look up")),
)


def heuristic_route(message: str) -> Optional[RouteDecision]:
    """Deterministic keyword classification. Returns None when nothing
    matches; the caller then routes to a direct answer."""
    text = message.lower()
    # "play <something>" as the very first word is a playback request.
    if text.startswith("play "):
        return RouteDecision(
            route=Route.SPOTIFY,
            reason="matched keyword for spotify",
            source="heuristic",
        )
    for route, keywords in _HEURISTIC_ROUTES:
        if any(k in text for k in keywords):
            return RouteDecision(
                route=route,
                reason=f"matched keyword for {route.value}",
                source="heuristic",
            )
    return None


def parse_route_decision(text: str) -> Optional[RouteDecision]:
    """Parse the model's routing answer. Returns None when the text is
    not a recognizable routing JSON object with a known route."""
    match = _JSON_BLOCK.search(text or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        route = Route(str(data.get("route", "")).strip().lower())
    except ValueError:
        return None
    reason = data.get("reason")
    return RouteDecision(
        route=route,
        reason=str(reason).strip() if reason else "",
        source="model",
    )


class Router:
    """Classifies each chat message into exactly one route.

    The same chat model that runs the agent loop does the routing, so
    the router is model-agnostic by construction: it works with the
    self-hosted Qwen3 server and with the hosted testing provider
    without code changes.
    """

    def __init__(self, settings: Settings, model: Optional[BaseChatModel] = None):
        self._model = model or build_chat_model(settings)

    def classify(self, message: str) -> RouteDecision:
        """Pick the route for one message. Never raises: model or parse
        trouble degrades to the keyword fallback, then to a direct
        answer, and the decision records which path chose it."""
        try:
            result = self._model.invoke(
                [
                    SystemMessage(content=ROUTING_PROMPT),
                    HumanMessage(content=message),
                ]
            )
            decision = parse_route_decision(
                result.content if isinstance(result.content, str) else str(result.content)
            )
            if decision is not None:
                return decision
            log.info("router: model answer not parseable, using keyword fallback")
        except Exception as exc:
            log.warning("router: model classification failed (%s), using keyword fallback", exc)
        decision = heuristic_route(message)
        if decision is not None:
            return decision
        return RouteDecision(
            route=Route.DIRECT_ANSWER,
            reason="no tool matches; answering directly",
            source="heuristic",
        )
