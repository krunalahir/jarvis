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
    PUBLISHED = "published"
    FAILED = "failed"


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
