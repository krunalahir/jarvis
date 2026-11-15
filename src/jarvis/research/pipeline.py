"""The research pipeline: question -> provider -> verified answer -> store.

The pipeline's job on top of the provider is honesty:
- no provider configured        -> NEEDS_CONFIGURATION
- provider fails                -> FAILED with the error
- no sources at all             -> UNCERTAIN
- claims without any source     -> moved to unsupported_claims, kept visible
- everything else               -> ANSWERED, saved to the store
"""

from __future__ import annotations

import logging

from ..schemas import AnswerStatus, ResearchAnswer
from .providers import ProviderError, ResearchProvider
from .store import Store

log = logging.getLogger(__name__)


class ResearchPipeline:
    def __init__(self, provider: ResearchProvider, store: Store):
        self.provider = provider
        self.store = store

    def ask(self, question: str) -> ResearchAnswer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        if not self.provider.available():
            hint = getattr(
                self.provider,
                "configuration_hint",
                "Set one of OPENAI_API_KEY, PERPLEXITY_API_KEY, EXA_API_KEY.",
            )
            answer = ResearchAnswer(
                question=question,
                status=AnswerStatus.NEEDS_CONFIGURATION,
                detail=f"Provider '{self.provider.name}' is not configured. {hint}",
            )
            self.store.save_answer(answer)
            return answer

        try:
            result = self.provider.research(question)
        except ProviderError as exc:
            log.warning("research provider failed: %s", exc)
            answer = ResearchAnswer(
                question=question, status=AnswerStatus.FAILED, detail=str(exc)
            )
            self.store.save_answer(answer)
            return answer

        # Citation validation: a source must be a real URL, and every claim's
        # source indexes must point at a source that actually exists. Bad
        # citations demote the claim to unsupported - links are never treated
        # as evidence just because they were attached.
        sources, remap = [], {}
        for old_index, source in enumerate(result.sources):
            if source.url and source.url.startswith(("http://", "https://")):
                remap[old_index] = len(sources)
                sources.append(source)

        supported, unsupported = [], []
        for claim in result.claims:
            claim.source_indexes = sorted(
                {remap[i] for i in claim.source_indexes if i in remap}
            )
            (supported if claim.source_indexes else unsupported).append(claim)

        if not sources:
            status = AnswerStatus.UNCERTAIN
            detail = "The provider returned no usable sources, so this answer cannot be verified."
        elif unsupported:
            status = AnswerStatus.UNCERTAIN
            detail = "Some statements have no supporting source and are listed separately."
        else:
            status = AnswerStatus.ANSWERED
            detail = ""

        answer = ResearchAnswer(
            question=question,
            status=status,
            summary=result.summary,
            claims=supported,
            sources=sources,
            unsupported_claims=[c.text for c in unsupported],
            detail=detail,
        )
        self.store.save_answer(answer)
        return answer
