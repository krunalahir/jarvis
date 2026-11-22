"""JARVIS HTTP API.

Run locally: uvicorn jarvis.api:app --host 127.0.0.1 --reload

Security model:
- Set JARVIS_API_KEY: every request then needs the X-API-Key header.
- JARVIS_ACTIONS_ENABLED=true unlocks capability only. It is not consent:
  every send/post/call/play/publish creates a pending action that the
  user must approve individually via POST /actions/{id}/approve.
- Refusing to start with actions enabled but no API key is deliberate;
  JARVIS_ALLOW_INSECURE=true overrides for local-only demos.
- Google sign-in starts in chat (the agent's connect_google tool). The
  only sign-in route is /auth/google/callback, where Google sends the
  user's browser afterwards. It takes no API key, because a browser
  redirect cannot send one; a one-time `state` value protects it instead.
"""

from __future__ import annotations

import html
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .actions import ActionGate
from .agent import build_agent, build_tools, run_agent
from .config import Settings
from .connectors import build_connectors
from .content import ContentGenerationUnavailable, ContentGenerator
from .google_oauth import GoogleOAuth, GoogleSignInError
from .llm import describe_llm
from .memory import MemoryManager
from .research import ResearchPipeline, Store, provider_from_settings
from .research.jobs import JobRunner
from .scheduler import PublishScheduler
from .schemas import ActionKind, AgentRunStatus, Platform

log = logging.getLogger(__name__)


class AskRequest(BaseModel):
    question: str


class DraftRequest(BaseModel):
    platform: Platform
    text: str
    media_urls: list[str] = []


class GenerateRequest(BaseModel):
    platform: Platform
    topic: str
    tone: Optional[str] = None


class ScheduleRequest(BaseModel):
    run_at: datetime


class EmailRequest(BaseModel):
    to: str
    subject: str
    body: str
    cc: list[str] = []


class EventRequest(BaseModel):
    summary: str
    start_iso: str
    end_iso: str
    attendees: list[str] = []
    description: str = ""


class CallRequest(BaseModel):
    to: str
    say: str


class PlayRequest(BaseModel):
    track_query: Optional[str] = None
    device_id: Optional[str] = None


class AgentChatRequest(BaseModel):
    message: str
    # Send a conversation_id and JARVIS remembers the conversation itself
    # (summary + recent turns + exact facts); then only the new message is
    # needed. Without it, the old stateless behaviour applies: optional
    # prior turns as {"role": "user"|"assistant", "content": str}.
    conversation_id: Optional[str] = None
    history: list[dict] = []


def _sign_in_page(ok: bool, message: str, status_code: int = 200) -> HTMLResponse:
    """The small page the user sees in the browser after Google sign-in."""
    title = "Google is connected" if ok else "Google was not connected"
    page = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>JARVIS - {title}</title></head>"
        "<body style='font-family: sans-serif; max-width: 34rem; margin: 4rem auto;'>"
        f"<h1>{title}</h1><p>{html.escape(message)}</p></body></html>"
    )
    return HTMLResponse(content=page, status_code=status_code)


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[ResearchPipeline] = None,
    store: Optional[Store] = None,
    connectors: Optional[dict] = None,
    content_generator: Optional[ContentGenerator] = None,
    agent_builder=None,
    google_oauth: Optional[GoogleOAuth] = None,
    memory: Optional[MemoryManager] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.actions_enabled and not settings.api_key and not settings.allow_insecure:
        raise RuntimeError(
            "Refusing to start: JARVIS_ACTIONS_ENABLED=true without JARVIS_API_KEY. "
            "Set an API key, or set JARVIS_ALLOW_INSECURE=true for local-only use."
        )
    store = store or Store(settings.store_path)
    pipeline = pipeline or ResearchPipeline(provider_from_settings(settings), store)
    # One Google sign-in for the whole app: the agent's connect_google tool
    # starts it, /auth/google/callback finishes it, and the Gmail, Calendar
    # and YouTube connectors take their access token from it.
    google_oauth = google_oauth or GoogleOAuth(settings)
    connectors = connectors or build_connectors(settings, google_oauth=google_oauth)
    scheduler = PublishScheduler(store, connectors)
    jobs = JobRunner(pipeline, store)
    generator = content_generator or ContentGenerator(settings)
    memory = memory or MemoryManager(store, settings)

    def _publish_single(draft_id: str) -> dict:
        draft = store.get_draft(draft_id)
        if draft is None:
            raise KeyError(f"unknown draft: {draft_id}")
        from .scheduler import PUBLISHERS

        connector_cls = PUBLISHERS.get(draft.platform)
        connector = connectors.get(connector_cls.name) if connector_cls else None
        if connector is None:
            raise ValueError(f"no publisher for {draft.platform}")
        return connector.publish_draft(draft)

    gate = ActionGate(
        store,
        {
            ActionKind.SEND_EMAIL: lambda p: connectors["gmail"].send_email(**p),
            ActionKind.CREATE_EVENT: lambda p: connectors["google_calendar"].create_event(**p),
            ActionKind.START_CALL: lambda p: connectors["calling"].start_call(**p),
            ActionKind.SPOTIFY_PLAY: lambda p: connectors["spotify"].play(**p),
            ActionKind.PUBLISH_DRAFT: lambda p: _publish_single(p["draft_id"]),
        },
    )

    app = FastAPI(title="JARVIS", version="0.1.0")

    def check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    # The test chat page. No API key here: a browser opening a page cannot
    # send one. The page asks for the key and sends it with every request.
    @app.get("/", response_class=HTMLResponse)
    def chat_page() -> HTMLResponse:
        page = Path(__file__).parent / "web" / "chat.html"
        return HTMLResponse(content=page.read_text(encoding="utf-8"))

    @app.get("/health")
    def health() -> dict:
        return {
            "ok": True,
            "actions_enabled": settings.actions_enabled,
            "llm": describe_llm(settings),
            "security_warnings": settings.security_warnings(),
        }

    # --- agent: the LangGraph plan/tool/observe loop ---

    @app.post("/agent/chat", dependencies=[Depends(check_api_key)])
    def agent_chat(request: AgentChatRequest) -> dict:
        # Tools and the pending-action list are built per request, so one
        # chat turn can never see or claim another turn's queued actions.
        pending: list = []
        sign_in_links: list = []
        state = None
        history = request.history
        if request.conversation_id:
            try:
                state = memory.load(request.conversation_id)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            history = memory.context_messages(state)
        tools = build_tools(
            settings,
            pipeline,
            gate,
            scheduler,
            generator,
            connectors,
            pending_action_ids=pending,
            google_oauth=google_oauth,
            sign_in_links=sign_in_links,
        )
        builder = agent_builder or (lambda tool_list: build_agent(settings, tool_list))
        reply = run_agent(
            builder(tools),
            request.message,
            max_steps=settings.agent_max_steps,
            history=history,
            pending_action_ids=pending,
        )
        if sign_in_links:
            # The exact link, straight from the tool. A long URL repeated by
            # the model can come out damaged; this copy cannot.
            reply.google_sign_in_url = sign_in_links[-1]
        if state is not None:
            reply.conversation_id = state.id
            if reply.status != AgentRunStatus.FAILED:
                remembered = reply.reply
                if pending and not settings.auto_approve:
                    remembered += (
                        "\n[Queued for the user's approval, not executed yet: "
                        + ", ".join(pending) + "]"
                    )
                try:
                    memory.record(state, request.message.strip(), remembered)
                except Exception as exc:  # never lose the reply over memory
                    log.warning("could not save conversation memory: %s", exc)
        return reply.model_dump(mode="json")

    @app.get("/conversations/{conversation_id}", dependencies=[Depends(check_api_key)])
    def get_conversation(conversation_id: str) -> dict:
        try:
            state = memory.load(conversation_id)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        if not state.turns and not state.summary:
            raise HTTPException(status_code=404, detail="no memory for this conversation yet")
        return state.model_dump(mode="json")

    # --- Google sign-in ---
    # Sign-in is started in chat, never by calling an endpoint. This is the
    # one route it needs: after the user presses Allow, Google sends their
    # browser here with a one-time code. Nobody calls it by hand.

    @app.get("/auth/google/callback", response_class=HTMLResponse)
    def google_auth_callback(
        code: Optional[str] = None,
        state: Optional[str] = None,
        error: Optional[str] = None,
    ) -> HTMLResponse:
        if error or not code or not state:
            reason = (
                f"Google reported: {error}."
                if error
                else "The sign-in link came back without its code."
            )
            return _sign_in_page(
                False, f"{reason} Ask JARVIS to connect Google again.", 400
            )
        try:
            status = google_oauth.finish(code, state)
        except GoogleSignInError as exc:
            return _sign_in_page(False, str(exc), 400)
        except Exception as exc:  # Google refused the code, network down...
            log.warning("google sign-in failed: %s", exc)
            return _sign_in_page(False, f"Google did not accept the sign-in: {exc}", 502)
        who = f" as {status['email']}" if status.get("email") else ""
        message = f"Signed in{who}. You can close this tab and go back to JARVIS."
        if status["missing_scopes"]:
            message += (
                " Some permissions were left unticked, so parts of Gmail, "
                "Calendar or YouTube will not work: "
                + ", ".join(status["missing_scopes"])
            )
        return _sign_in_page(True, message)

    # --- research ---

    @app.post("/research/ask", dependencies=[Depends(check_api_key)])
    def ask(request: AskRequest) -> dict:
        return pipeline.ask(request.question).model_dump(mode="json")

    @app.post("/research/jobs", dependencies=[Depends(check_api_key)], status_code=202)
    def submit_job(request: AskRequest) -> dict:
        return jobs.submit(request.question).model_dump(mode="json")

    @app.get("/research/jobs/{job_id}", dependencies=[Depends(check_api_key)])
    def get_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job.model_dump(mode="json")

    @app.get("/research/{answer_id}", dependencies=[Depends(check_api_key)])
    def get_answer(answer_id: str) -> dict:
        answer = store.get_answer(answer_id)
        if answer is None:
            raise HTTPException(status_code=404, detail="answer not found")
        return answer.model_dump(mode="json")

    @app.get("/research", dependencies=[Depends(check_api_key)])
    def list_answers(limit: int = 50) -> list[dict]:
        return [a.model_dump(mode="json") for a in store.list_answers(limit)]

    # --- connectors ---

    @app.get("/connectors/status", dependencies=[Depends(check_api_key)])
    def connector_status() -> list[dict]:
        return [c.status() for c in connectors.values()]

    # --- content ---

    @app.post("/content/generate", dependencies=[Depends(check_api_key)])
    def generate_content(request: GenerateRequest) -> dict:
        try:
            text = generator.generate(request.platform, request.topic, request.tone)
        except ContentGenerationUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        draft = scheduler.create_draft(request.platform, text)
        return draft.model_dump(mode="json")

    @app.post("/content/drafts", dependencies=[Depends(check_api_key)])
    def create_draft(request: DraftRequest) -> dict:
        draft = scheduler.create_draft(request.platform, request.text, request.media_urls)
        return draft.model_dump(mode="json")

    @app.post("/content/drafts/{draft_id}/schedule", dependencies=[Depends(check_api_key)])
    def schedule_draft(draft_id: str, request: ScheduleRequest) -> dict:
        try:
            draft = scheduler.schedule(draft_id, request.run_at)
        except KeyError:
            raise HTTPException(status_code=404, detail="draft not found")
        return draft.model_dump(mode="json")

    @app.post("/content/publish-due", dependencies=[Depends(check_api_key)])
    def publish_due() -> list[dict]:
        return [d.model_dump(mode="json") for d in scheduler.publish_due()]

    # --- actions: every real-world effect is requested, then approved ---

    @app.post("/gmail/send", dependencies=[Depends(check_api_key)], status_code=202)
    def request_email(request: EmailRequest) -> dict:
        action = gate.request(
            ActionKind.SEND_EMAIL,
            f"Send email to {request.to}: '{request.subject}'",
            request.model_dump(),
        )
        return action.model_dump(mode="json")

    @app.post("/calendar/events", dependencies=[Depends(check_api_key)], status_code=202)
    def request_event(request: EventRequest) -> dict:
        action = gate.request(
            ActionKind.CREATE_EVENT,
            f"Create calendar event '{request.summary}' at {request.start_iso}",
            request.model_dump(),
        )
        return action.model_dump(mode="json")

    @app.post("/calls", dependencies=[Depends(check_api_key)], status_code=202)
    def request_call(request: CallRequest) -> dict:
        action = gate.request(
            ActionKind.START_CALL,
            f"Call {request.to} and say: '{request.say[:80]}'",
            request.model_dump(),
        )
        return action.model_dump(mode="json")

    @app.post("/spotify/play", dependencies=[Depends(check_api_key)], status_code=202)
    def request_play(request: PlayRequest) -> dict:
        action = gate.request(
            ActionKind.SPOTIFY_PLAY,
            f"Play on Spotify: {request.track_query or 'resume'}",
            request.model_dump(exclude_none=True),
        )
        return action.model_dump(mode="json")

    @app.post("/content/drafts/{draft_id}/publish-now", dependencies=[Depends(check_api_key)], status_code=202)
    def request_publish(draft_id: str) -> dict:
        draft = store.get_draft(draft_id)
        if draft is None:
            raise HTTPException(status_code=404, detail="draft not found")
        action = gate.request(
            ActionKind.PUBLISH_DRAFT,
            f"Publish {draft.platform.value} post: '{draft.text[:80]}'",
            {"draft_id": draft_id},
        )
        return action.model_dump(mode="json")

    @app.get("/actions", dependencies=[Depends(check_api_key)])
    def list_actions(limit: int = 50) -> list[dict]:
        return [a.model_dump(mode="json") for a in gate.list(limit)]

    @app.post("/actions/{action_id}/approve", dependencies=[Depends(check_api_key)])
    def approve_action(action_id: str) -> dict:
        try:
            return gate.approve(action_id).model_dump(mode="json")
        except KeyError:
            raise HTTPException(status_code=404, detail="action not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/actions/{action_id}/reject", dependencies=[Depends(check_api_key)])
    def reject_action(action_id: str) -> dict:
        try:
            return gate.reject(action_id).model_dump(mode="json")
        except KeyError:
            raise HTTPException(status_code=404, detail="action not found")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return app


app = create_app()
