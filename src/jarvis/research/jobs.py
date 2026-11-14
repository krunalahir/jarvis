"""Background research jobs.

Deep research can take minutes, so the API also exposes research as a
background job: submit a question, poll the job, get the answer id when
done. Answers are persisted in the Store as usual; job state is kept in
memory (restart loses pending jobs, never saved answers).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..schemas import JobStatus, ResearchJob, utcnow
from .pipeline import ResearchPipeline


class JobRunner:
    def __init__(self, pipeline: ResearchPipeline, workers: int = 2):
        self._pipeline = pipeline
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._jobs: dict[str, ResearchJob] = {}
        self._lock = threading.Lock()

    def submit(self, question: str) -> ResearchJob:
        job = ResearchJob(question=question.strip())
        if not job.question:
            raise ValueError("question must not be empty")
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> Optional[ResearchJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.status = JobStatus.RUNNING
        try:
            answer = self._pipeline.ask(job.question)
            with self._lock:
                job.status = JobStatus.DONE
                job.answer_id = answer.id
        except Exception as exc:  # job failure stays visible on the job
            with self._lock:
                job.status = JobStatus.FAILED
                job.error = str(exc)
        finally:
            with self._lock:
                job.finished_at = utcnow()
