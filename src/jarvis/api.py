"""JARVIS HTTP API.

Run: uvicorn jarvis.api:app --reload

If JARVIS_API_KEY is set, every request needs the header X-API-Key.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .config import Settings
from .connectors import build_connectors
from .research import ResearchPipeline, Store, provider_from_settings
from .research.jobs import JobRunner
from .scheduler import PublishScheduler
from .schemas import Platform


class AskRequest(BaseModel):
    question: str


class DraftRequest(BaseModel):
    platform: Platform
    text: str
    media_urls: list[str] = []


class ScheduleRequest(BaseModel):
    run_at: datetime


class PlayRequest(BaseModel):
    track_query: Optional[str] = None
    device_id: Optional[str] = None


def create_app(
    settings: Optional[Settings] = None,
    pipeline: Optional[ResearchPipeline] = None,
    store: Optional[Store] = None,
    connectors: Optional[dict] = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    store = store or Store(settings.store_path)
    pipeline = pipeline or ResearchPipeline(provider_from_settings(settings), store)
    connectors = connectors or build_connectors(settings)
    scheduler = PublishScheduler(store, connectors)
    jobs = JobRunner(pipeline)

    app = FastAPI(title="JARVIS", version="0.1.0")

    def check_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if settings.api_key and x_api_key != settings.api_key:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/health")
    def health() -> dict:
        return {"ok": True, "actions_enabled": settings.actions_enabled}

    @app.post("/research/ask", dependencies=[Depends(check_api_key)])
    def ask(request: AskRequest) -> dict:
        return pipeline.ask(request.question).model_dump(mode="json")

    @app.post("/research/jobs", dependencies=[Depends(check_api_key)])
    def submit_job(request: AskRequest) -> dict:
        """Submit research as a background job; poll GET /research/jobs/{id}."""
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

    @app.get("/connectors/status", dependencies=[Depends(check_api_key)])
    def connector_status() -> list[dict]:
        return [c.status() for c in connectors.values()]

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

    @app.post("/spotify/play", dependencies=[Depends(check_api_key)])
    def spotify_play(request: PlayRequest) -> dict:
        try:
            return connectors["spotify"].play(request.track_query, request.device_id)
        except Exception as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return app


app = create_app()
