"""Shared data models for JARVIS."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Source(BaseModel):
    title: str
    url: str
    publisher: Optional[str] = None
    published_at: Optional[str] = None
    snippet: Optional[str] = None


class Claim(BaseModel):
    """One statement in an answer, with the sources that support it.

    source_indexes point into the answer's sources list. A claim with no
    sources is unsupported and is surfaced as such, never silently kept.
    """

    text: str
    source_indexes: list[int] = Field(default_factory=list)


class AnswerStatus(str, Enum):
    ANSWERED = "answered"
    UNCERTAIN = "uncertain"  # answered, but with unsupported or missing evidence
    NEEDS_CONFIGURATION = "needs_configuration"  # no research provider set up
    FAILED = "failed"


class ResearchAnswer(BaseModel):
    id: str = Field(default_factory=new_id)
    question: str
    status: AnswerStatus
    summary: str = ""
    claims: list[Claim] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    detail: str = ""
    created_at: datetime = Field(default_factory=utcnow)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class ResearchJob(BaseModel):
    id: str = Field(default_factory=new_id)
    question: str
    status: JobStatus = JobStatus.PENDING
    answer_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: Optional[datetime] = None


class Platform(str, Enum):
    INSTAGRAM = "instagram"
    X = "x"
    YOUTUBE = "youtube"
    SNAPCHAT = "snapchat"  # unconfirmed interpretation of "snaptrack"


class DraftStatus(str, Enum):
    DRAFT = "draft"
    SCHEDULED = "scheduled"
    PUBLISHING = "publishing"  # claimed by a publisher run, not yet done
    PUBLISHED = "published"
    FAILED = "failed"


class ActionKind(str, Enum):
    SEND_EMAIL = "send_email"
    CREATE_EVENT = "create_event"
    START_CALL = "start_call"
    PUBLISH_DRAFT = "publish_draft"
    SPOTIFY_PLAY = "spotify_play"


class PendingActionStatus(str, Enum):
    PENDING = "pending"      # waiting for the user's explicit approval
    EXECUTING = "executing"  # claimed by one approve call; executor running
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"        # approved but execution failed; error is kept


class PendingAction(BaseModel):
    """A real-world action the assistant wants to take. Nothing executes
    until the user approves this specific action - the global
    JARVIS_ACTIONS_ENABLED switch only gates capability, not consent."""

    id: str = Field(default_factory=new_id)
    kind: ActionKind
    summary: str  # human-readable: "Send email to x@y.com: 'Subject'"
    payload: dict = Field(default_factory=dict)
    status: PendingActionStatus = PendingActionStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)
    decided_at: Optional[datetime] = None


class PostDraft(BaseModel):
    """A piece of social content. Creating a draft never publishes it."""

    id: str = Field(default_factory=new_id)
    platform: Platform
    text: str
    media_urls: list[str] = Field(default_factory=list)
    scheduled_at: Optional[datetime] = None
    status: DraftStatus = DraftStatus.DRAFT
    platform_post_id: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=utcnow)


class Route(str, Enum):
    """The one capability the router picks for a chat message.

    Every message gets exactly one route. DIRECT_ANSWER means no tool
    fits: the model answers from what it knows and calls nothing.
    """

    DIRECT_ANSWER = "direct_answer"
    RESEARCH = "research"
    WEB_SEARCH = "web_search"
    GMAIL = "gmail"
    CALENDAR = "calendar"
    PHONE_CALL = "phone_call"
    SPOTIFY = "spotify"
    SOCIAL_CONTENT = "social_content"
    APP_STATUS = "app_status"


class RouteDecision(BaseModel):
    """The router's pick for one message.

    source records how the route was chosen: "model" when the chat
    model's classification parsed cleanly, "heuristic" when the keyword
    fallback decided (model unparsable or unreachable).
    """

    route: Route
    reason: str = ""
    source: str = "model"


class TaskStage(str, Enum):
    """Stages the task monitor reports while one request runs."""

    RECEIVED = "received"
    ROUTED = "routed"
    AUTH_CHECK = "auth_check"
    EXECUTING = "executing"
    STEP = "step"
    APPROVAL_NEEDED = "approval_needed"
    RECOVERING = "recovering"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskUpdate(BaseModel):
    """One progress update for a running task, written for the chat.

    The monitor emits these from message receipt to completion (auth
    checks, tool steps, approval requests, recovery, outcome) so the
    chat client can show the user what JARVIS is doing while it works.
    """

    task_id: str
    seq: int
    stage: TaskStage
    message: str
    created_at: datetime = Field(default_factory=utcnow)


class AgentStep(BaseModel):
    """One tool the agent called while working."""

    tool: str
    arguments: dict = Field(default_factory=dict)


class AgentRunStatus(str, Enum):
    ANSWERED = "answered"
    STOPPED_STEP_LIMIT = "stopped_step_limit"  # loop bound hit; nothing was left running
    FAILED = "failed"


class AgentReply(BaseModel):
    """Result of one agent chat turn.

    pending_action_ids lists any write actions the agent queued through
    the approval gate. The agent can only request them; each still needs
    the user's explicit approval before anything executes.

    route is the router's pick for this message; task_id and updates are
    the monitor's progress trail from receipt to outcome.
    """

    reply: str
    status: AgentRunStatus
    steps: list[AgentStep] = Field(default_factory=list)
    pending_action_ids: list[str] = Field(default_factory=list)
    detail: str = ""
    route: Optional[Route] = None
    task_id: Optional[str] = None
    updates: list[TaskUpdate] = Field(default_factory=list)

