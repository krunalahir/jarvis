"""Background research jobs.

Deep research can take minutes, so the API also exposes research as a
background job: submit a question, poll the job, get the answer id when
done. Jobs and answers are persisted in the Store, so a restart never
loses a finished answer; jobs that were pending or running at shutdown
are requeued on the next start.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from ..schemas import JobStatus, ResearchJob, utcnow
from .pipeline import ResearchPipeline
from .store import Store


class JobRunner:
    def __init__(self, pipeline: ResearchPipeline, store: Store, workers: int = 2):
        self._pipeline = pipeline
        self._store = store
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._lock = threading.Lock()
        # Restart recovery: anything left pending/running was interrupted.
        for job in self._store.interrupted_jobs():
            job.status = JobStatus.PENDING
            job.error = None
            self._store.save_job(job)
            self._pool.submit(self._run, job.id)

    def submit(self, question: str) -> ResearchJob:
        job = ResearchJob(question=question.strip())
        if not job.question:
            raise ValueError("question must not be empty")
        self._store.save_job(job)
        self._pool.submit(self._run, job.id)
        return job

    def get(self, job_id: str) -> Optional[ResearchJob]:
        return self._store.get_job(job_id)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _run(self, job_id: str) -> None:
        with self._lock:
            job = self._store.get_job(job_id)
            if job is None:
                return
            job.status = JobStatus.RUNNING
            self._store.save_job(job)
        try:
            answer = self._pipeline.ask(job.question)
            job.status = JobStatus.DONE
            job.answer_id = answer.id
        except Exception as exc:  # job failure stays visible on the job
            job.status = JobStatus.FAILED
            job.error = str(exc)
        finally:
            job.finished_at = utcnow()
            self._store.save_job(job)
