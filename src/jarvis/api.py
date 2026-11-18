"""JARVIS HTTP API.

Run locally: uvicorn jarvis.api:app --host 127.0.0.1 --reload

Security model:
- Set JARVIS_API_KEY: every request then needs the X-API-Key header.
- JARVIS_ACTIONS_ENABLED=true unlocks capability only. It is not consent:
  every send/post/call/play/publish creates a pending action that the
  user must approve individually via POST /actions/{id}/approve.
- Refusing to start with actions enabled but no API key is deliberate;
  JARVIS_ALLOW_INSECURE=true overrides for local-only demos.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, field_validator

from .actions import ActionGate
from .agent import build_agent, build_tools, run_task
from .config import Settings
from .connectors import build_connectors
from .content import ContentGenerationUnavailable, ContentGenerator
from .llm import describe_llm
from .recovery import recover_interrupted_state
from .research import ResearchPipeline, Store, provider_from_settings
from .research.jobs import JobRunner
from .scheduler import PublishScheduler
from .schemas import ActionKind, Platform


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
    # Optional prior turns as {"role": "user"|"assistant", "content": str}.
    # Validated so malformed or oversized histories fail with 422 instead
    # of breaking the agent run or overflowing the model's context window.
    history: list[dict] = []

    @field_validator("history")
    @classmethod
    def _validate_history(cls, value: list[dict]) -> list[dict]:
        if len(value) > 50:
            raise ValueError("history is limited to 50 messages")
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("history entries must be objects with role and content")
            if item.get("role") not in ("user", "assistant"):
                raise ValueError("history roles must be 'user' or 'assistant'")
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                raise ValueError("history content must be non-empty text")
            if len(content) > 4000:
                raise ValueError("history messages are limited to 4000 characters")
        return value


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[ResearchPipeline] = None,
    store: Optional[Store] = None,
    connectors: Optional[dict] = None,
    content_generator: Optional[ContentGenerator] = None,
    agent_builder=None,
    router=None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    if settings.actions_enabled and not settings.api_key and not settings.allow_insecure:
        raise RuntimeError(
            "Refusing to start: JARVIS_ACTIONS_ENABLED=true without JARVIS_API_KEY. "
            "Set an API key, or set JARVIS_ALLOW_INSECURE=true for local-only use."
        )
    store = store or Store(settings.store_path)
    # Crash recovery: anything stuck mid-execution/mid-publish is marked
    # failed-unknown and surfaced for review. Never retried - the external
    # effect may already have happened.
    recover_interrupted_state(store)
    pipeline = pipeline or ResearchPipeline(provider_from_settings(settings), store)
    connectors = connectors or build_connectors(settings)
    scheduler = PublishScheduler(store, connectors)
    jobs = JobRunner(pipeline, store)
    generator = content_generator or ContentGenerator(settings)

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
        # One chat entry point for everything: the router reads the
        # message, picks the route (research, an app, or a direct
        # answer), a bounded agent loop executes it with that route's
        # tools, and the monitor supervises the run end to end. The
        # reply carries the route and the progress updates.
        reply = run_task(
            settings,
            request.message,
            history=request.history,
            max_steps=settings.agent_max_steps,
            store=store,
            pipeline=pipeline,
            gate=gate,
            scheduler=scheduler,
            generator=generator,
            connectors=connectors,
            agent_builder=(
                (lambda tool_list: agent_builder(tool_list)) if agent_builder else None
            ),
            router=router,
        )
        return reply.model_dump(mode="json")

    @app.get("/agent/tasks/{task_id}/updates", dependencies=[Depends(check_api_key)])
    def task_updates(task_id: str) -> list[dict]:
        updates = store.list_task_updates(task_id)
        if not updates:
            raise HTTPException(status_code=404, detail="task not found")
        return [u.model_dump(mode="json") for u in updates]

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

