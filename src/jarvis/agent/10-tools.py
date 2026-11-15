"""Agent tools: what the LangGraph agent is allowed to do.

Read tools (search, research, reads through connectors) run directly.
Write tools NEVER execute anything: they create a PendingAction through
the ActionGate, exactly like the HTTP endpoints do, and the user approves
each one explicitly. The agent therefore cannot send, post, call, play or
publish on its own, no matter what the model decides.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ..actions import ActionGate
from ..config import Settings
from ..connectors.base import ConnectorError
from ..content import ContentGenerationUnavailable, ContentGenerator
from ..research.pipeline import ResearchPipeline
from ..research.providers import ProviderError, exa_search_sources
from ..scheduler import PublishScheduler
from ..schemas import ActionKind, Platform


def build_tools(
    settings: Settings,
    pipeline: ResearchPipeline,
    gate: ActionGate,
    scheduler: PublishScheduler,
    generator: ContentGenerator,
    connectors: dict,
    pending_action_ids: Optional[list] = None,
) -> list:
    """Build the agent's tool list. `pending_action_ids`, when given,
    collects the id of every action the agent queues, so the caller can
    report them back to the user."""

    def _record(action) -> str:
        if pending_action_ids is not None:
            pending_action_ids.append(action.id)
        return (
            f"Queued for user approval as pending action {action.id}: "
            f"{action.summary}. Nothing has been executed; the user must "
            "approve this specific action before it happens."
        )

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Returns numbered sources
        with titles, URLs and snippets. Use before answering questions about
        recent or external facts."""
        if not settings.exa_api_key:
            return (
                "Web search is not configured: EXA_API_KEY is missing. The "
                "local model cannot browse the web by itself, so do not "
                "invent current facts; tell the user what is needed."
            )
        try:
            sources = exa_search_sources(settings.exa_api_key, query)
        except ProviderError as exc:
            return f"Web search failed: {exc}"
        if not sources:
            return "No results found."
        return "\n".join(
            f"[{i}] {s.title} - {s.url}\n{s.snippet or ''}"
            for i, s in enumerate(sources, start=1)
        )

    @tool
    def research(question: str) -> str:
        """Run the full research pipeline: web search plus a cited answer
        from the local model, saved to the store. Prefer this for questions
        that need depth and sources."""
        answer = pipeline.ask(question)
        parts = [f"status: {answer.status.value}"]
        if answer.summary:
            parts.append(f"summary: {answer.summary}")
        if answer.sources:
            parts.append(
                "sources:\n"
                + "\n".join(
                    f"[{i}] {s.title} - {s.url}"
                    for i, s in enumerate(answer.sources, start=1)
                )
            )
        if answer.detail:
            parts.append(f"detail: {answer.detail}")
        parts.append(f"answer_id: {answer.id}")
        return "\n".join(parts)

    @tool
    def draft_post(platform: str, topic: str, tone: Optional[str] = None) -> str:
        """Write a social post draft with the local model and save it as a
        draft. This NEVER publishes; publishing is a separate approved
        action."""
        try:
            plat = Platform(platform.strip().lower())
        except ValueError:
            return f"Unknown platform {platform!r}; expected one of: instagram, x, youtube, snapchat."
        try:
            text = generator.generate(plat, topic, tone)
        except ContentGenerationUnavailable as exc:
            return f"Draft not created: {exc}"
        draft = scheduler.create_draft(plat, text)
        return f"Draft {draft.id} saved (never publishes itself):\n{text}"

    @tool
    def connector_status() -> str:
        """List every connected app and whether it is configured and ready."""
        return "\n".join(
            f"{c['name']}: {c['state']}"
            + (f" (missing: {', '.join(c['missing_settings'])})" if c["missing_settings"] else "")
            for c in (conn.status() for conn in connectors.values())
        )

    @tool
    def read_emails(query: str, limit: int = 5) -> str:
        """Read-only Gmail search that fetches each matching message, so the
        result carries the sender, subject, date and body text. Needs
        GOOGLE_ACCESS_TOKEN; sends nothing."""
        gmail = connectors["gmail"]
        try:
            messages = gmail.read_messages(query, limit=limit)
        except ConnectorError as exc:
            return str(exc)
        if not messages:
            return "No matching messages."
        return "\n\n".join(
            f"From: {m['from']}\nSubject: {m['subject']}\nDate: {m['date']}\n"
            f"{(m['body'] or m['snippet'])[:500]}"
            for m in messages
        )

    @tool
    def list_calendar_events(limit: int = 5) -> str:
        """Read-only list of upcoming calendar events. Needs
        GOOGLE_ACCESS_TOKEN; creates nothing."""
        calendar = connectors["google_calendar"]
        try:
            events = calendar.list_events(limit=limit)
        except ConnectorError as exc:
            return str(exc)
        if not events:
            return "No upcoming events."
        return "\n".join(
            f"{e.get('summary', '(no title)')} at {e.get('start', {})}" for e in events
        )

    @tool
    def request_send_email(to: str, subject: str, body: str) -> str:
        """Ask to send an email. This only QUEUES a pending action for the
        user to approve; it never sends by itself."""
        action = gate.request(
            ActionKind.SEND_EMAIL,
            f"Send email to {to}: '{subject}'",
            {"to": to, "subject": subject, "body": body, "cc": []},
        )
        return _record(action)

    @tool
    def request_create_event(summary: str, start_iso: str, end_iso: str) -> str:
        """Ask to create a calendar event. This only QUEUES a pending action
        for the user to approve; it never creates by itself."""
        action = gate.request(
            ActionKind.CREATE_EVENT,
            f"Create calendar event '{summary}' at {start_iso}",
            {"summary": summary, "start_iso": start_iso, "end_iso": end_iso, "attendees": [], "description": ""},
        )
        return _record(action)

    @tool
    def request_phone_call(to: str, say: str) -> str:
        """Ask to start a phone call. This only QUEUES a pending action for
        the user to approve; it never calls by itself."""
        action = gate.request(
            ActionKind.START_CALL,
            f"Call {to} and say: '{say[:80]}'",
            {"to": to, "say": say},
        )
        return _record(action)

    @tool
    def request_spotify_play(track_query: Optional[str] = None) -> str:
        """Ask to play music on Spotify. This only QUEUES a pending action
        for the user to approve; it never plays by itself."""
        action = gate.request(
            ActionKind.SPOTIFY_PLAY,
            f"Play on Spotify: {track_query or 'resume'}",
            {k: v for k, v in {"track_query": track_query}.items() if v},
        )
        return _record(action)

    @tool
    def request_publish_draft(draft_id: str) -> str:
        """Ask to publish a saved post draft now. This only QUEUES a pending
        action for the user to approve; it never publishes by itself."""
        action = gate.request(
            ActionKind.PUBLISH_DRAFT,
            f"Publish draft {draft_id}",
            {"draft_id": draft_id},
        )
        return _record(action)

    return [
        web_search,
        research,
        draft_post,
        connector_status,
        read_emails,
        list_calendar_events,
        request_send_email,
        request_create_event,
        request_phone_call,
        request_spotify_play,
        request_publish_draft,
    ]

