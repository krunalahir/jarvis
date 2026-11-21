"""Agent tools: what the LangGraph agent is allowed to do.

Read tools (search, research, reads through connectors) run directly.
Write tools NEVER execute anything: they create a PendingAction through
the ActionGate, exactly like the HTTP endpoints do, and the user approves
each one explicitly. The agent therefore cannot send, post, call, play or
publish on its own, no matter what the model decides.

Google sign-in also starts here: connect_google hands the user a link.
The agent cannot sign in for the user; only the user can open the link
and press Allow.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import tool

from ..actions import ActionGate
from ..config import Settings
from ..connectors.base import ConnectorError
from ..content import ContentGenerationUnavailable, ContentGenerator
from ..google_oauth import SIGN_IN_LINK_TTL_SECONDS
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
    google_oauth=None,
    sign_in_links: Optional[list] = None,
) -> list:
    """Build the agent's tool list.

    `pending_action_ids`, when given, collects the id of every action the
    agent queues, so the caller can report them back to the user.
    `google_oauth` is the app's Google sign-in (jarvis.google_oauth).
    `sign_in_links`, when given, collects every sign-in link handed out,
    so the caller can show the exact link even if the model miscopies it.
    """

    def _record(action) -> str:
        if pending_action_ids is not None:
            pending_action_ids.append(action.id)
        if settings.auto_approve:
            done = gate.approve(action.id)
            if done.status.value == "executed":
                return f"Done, this was executed now: {action.summary}."
            return f"This FAILED and was not done: {action.summary}. Error: {done.error}"
        return (
            f"Queued for user approval as pending action {action.id}: "
            f"{action.summary}. Nothing has been executed; the user must "
            "approve this specific action before it happens."
        )

    def _google_note(connector_name: str) -> str:
        """Extra line for a queued Google action when nobody is signed in."""
        connector = connectors.get(connector_name)
        if connector is None or connector.configured():
            return ""
        return (
            " Google is not connected yet, so this would fail if approved "
            "now. Call connect_google and give the user the sign-in link; "
            "they should approve after signing in."
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
    def connect_google(reconnect: bool = False) -> str:
        """Connect the user's Google account so Gmail, Calendar and YouTube
        work. Returns a sign-in link that the user must open; only the user
        can press Allow. Use this when the user asks to sign in to or
        connect Google, or when another tool says Google is not connected.
        Set reconnect to true only if the user wants to switch account or
        grant permissions again."""
        if google_oauth is None:
            return "Google sign-in is not available in this setup."
        problem = google_oauth.problem()
        if problem:
            return (
                f"Google sign-in is not set up yet: {problem} Tell the user "
                "this; there is no link to give them."
            )
        status = google_oauth.status()
        if status["connected"] and not status["missing_scopes"] and not reconnect:
            who = f" as {status['email']}" if status.get("email") else ""
            return f"Google is already connected{who}. No sign-in is needed."
        try:
            link = google_oauth.start()
        except Exception as exc:
            return f"Could not start Google sign-in: {exc}"
        if sign_in_links is not None:
            sign_in_links.append(link)
        minutes = SIGN_IN_LINK_TTL_SECONDS // 60
        return (
            "Google is NOT connected yet. Give the user this link, copied "
            "exactly, and ask them to open it in their browser, choose "
            "their Google account and press Allow. The link works for "
            f"{minutes} minutes. Do not use Gmail or Calendar tools again "
            "until the user says they have finished.\n"
            f"Sign-in link: {link}"
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
            + (f" - {c['note']}" if c.get("note") else "")
            for c in (conn.status() for conn in connectors.values())
        )

    @tool
    def read_emails(query: str, limit: int = 5) -> str:
        """Read-only Gmail search. Needs Google to be connected (see
        connect_google); returns the ids of matching messages, sends
        nothing."""
        gmail = connectors["gmail"]
        try:
            messages = gmail.search_messages(query, limit=limit)
        except ConnectorError as exc:
            return str(exc)
        if not messages:
            return "No matching messages."
        return "\n".join(str(m) for m in messages)

    @tool
    def list_calendar_events(limit: int = 5) -> str:
        """Read-only list of upcoming calendar events. Needs Google to be
        connected (see connect_google); creates nothing."""
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
        """       if "[" in body and "]" in body:
            return (
                "Not sent: the body still contains a placeholder in square "
                "brackets. Ask the user for the missing name or detail, then "
                "call this tool again with the real text."
            )
        Ask to send an email. This only QUEUES a pending action for the
        user to approve; it never sends by itself."""
        action = gate.request(
            ActionKind.SEND_EMAIL,
            f"Send email to {to}: '{subject}'",
            {"to": to, "subject": subject, "body": body, "cc": []},
        )
        return _record(action) + _google_note("gmail")

    @tool
    def request_create_event(summary: str, start_iso: str, end_iso: str) -> str:
        """Ask to create a calendar event. This only QUEUES a pending action
        for the user to approve; it never creates by itself."""
        action = gate.request(
            ActionKind.CREATE_EVENT,
            f"Create calendar event '{summary}' at {start_iso}",
            {"summary": summary, "start_iso": start_iso, "end_iso": end_iso, "attendees": [], "description": ""},
        )
        return _record(action) + _google_note("google_calendar")

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
        connect_google,
        read_emails,
        list_calendar_events,
        request_send_email,
        request_create_event,
        request_phone_call,
        request_spotify_play,
        request_publish_draft,
    ]
