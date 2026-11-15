"""Research providers.

A provider answers a question and returns a summary plus sources, with
claims linked to the sources that support them. The default provider is
`LocalQwenResearchProvider`: a self-hosted Qwen3 8B (through LangChain)
writes the answer, and a separate web search API (Exa) fetches the
sources - the model never has to invent facts it cannot see. Hosted
providers (OpenAI, Perplexity, Exa-only) remain available as explicit
opt-ins via JARVIS_RESEARCH_PROVIDER. `StaticProvider` exists for tests
and demos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from ..config import Settings
from ..llm import build_chat_model
from ..schemas import Claim, Source


@dataclass
class ProviderResult:
    summary: str = ""
    claims: list[Claim] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)


class ResearchProvider(Protocol):
    name: str
    configuration_hint: str

    def available(self) -> bool: ...

    def research(self, question: str) -> ProviderResult: ...


class ProviderError(Exception):
    pass


_CITATION_RE = re.compile(r"\[(\d+)\]")


def claims_from_cited_text(text: str, source_count: int) -> list[Claim]:
    """Split provider text into sentences and link [n] citations to sources.

    Providers like Perplexity return answers with inline [1] [2] markers.
    Sentences without any marker become claims with no sources, which the
    pipeline surfaces as unsupported rather than dropping silently.
    """
    # A sentence plus any citation markers that follow its final period.
    sentences = re.findall(r"[^.!?]+[.!?]+(?:\s*\[\d+\])*|[^.!?]+$", text)
    claims: list[Claim] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        indexes = [int(m) - 1 for m in _CITATION_RE.findall(sentence)]
        indexes = [i for i in indexes if 0 <= i < source_count]
        clean = _CITATION_RE.sub("", sentence).strip()
        if clean:
            claims.append(Claim(text=clean, source_indexes=sorted(set(indexes))))
    return claims


class OpenAIWebSearchProvider:
    """OpenAI Responses API with the web_search tool."""

    name = "openai"
    configuration_hint = "Set OPENAI_API_KEY, or use the default local Qwen3 provider."
    endpoint = "https://api.openai.com/v1/responses"

    def __init__(self, api_key: Optional[str], client: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def available(self) -> bool:
        return bool(self._api_key)

    def research(self, question: str) -> ProviderResult:
        response = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": "gpt-4.1",
                "tools": [{"type": "web_search_preview"}],
                "input": question,
            },
        )
        if response.status_code != 200:
            raise ProviderError(f"OpenAI API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        text, sources = "", []
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text += part.get("text", "")
                    for ann in part.get("annotations", []):
                        if ann.get("type") == "url_citation":
                            sources.append(
                                Source(title=ann.get("title", ""), url=ann.get("url", ""))
                            )
        return ProviderResult(
            summary=text,
            claims=claims_from_cited_text(text, len(sources)),
            sources=sources,
        )


class PerplexityProvider:
    """Perplexity chat completions (sonar), which returns citations."""

    name = "perplexity"
    configuration_hint = "Set PERPLEXITY_API_KEY, or use the default local Qwen3 provider."
    endpoint = "https://api.perplexity.ai/chat/completions"

    def __init__(self, api_key: Optional[str], client: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def available(self) -> bool:
        return bool(self._api_key)

    def research(self, question: str) -> ProviderResult:
        response = self._client.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": "sonar",
                "messages": [{"role": "user", "content": question}],
            },
        )
        if response.status_code != 200:
            raise ProviderError(f"Perplexity API error {response.status_code}: {response.text[:200]}")
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        sources = [Source(title=u, url=u) for u in data.get("citations", [])]
        return ProviderResult(
            summary=text,
            claims=claims_from_cited_text(text, len(sources)),
            sources=sources,
        )


class ExaProvider:
    """Exa search: returns sources only, so the summary stays empty and the
    pipeline reports what was found per source instead of inventing text."""

    name = "exa"
    configuration_hint = "Set EXA_API_KEY, or use the default local Qwen3 provider."
    endpoint = "https://api.exa.ai/search"

    def __init__(self, api_key: Optional[str], client: Optional[httpx.Client] = None):
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=60)

    def available(self) -> bool:
        return bool(self._api_key)

    def research(self, question: str) -> ProviderResult:
        response = self._client.post(
            self.endpoint,
            headers={"x-api-key": self._api_key or ""},
            json={"query": question, "numResults": 8, "contents": {"text": True}},
        )
        if response.status_code != 200:
            raise ProviderError(f"Exa API error {response.status_code}: {response.text[:200]}")
        sources = []
        claims = []
        for i, r in enumerate(response.json().get("results", [])):
            sources.append(
                Source(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    published_at=r.get("publishedDate"),
                    snippet=(r.get("text") or "")[:500],
                )
            )
            if r.get("text"):
                claims.append(Claim(text=r["text"][:500], source_indexes=[i]))
        return ProviderResult(summary="", claims=claims, sources=sources)


class StaticProvider:
    """Fixed responses for tests and offline demos."""

    name = "static"

    def __init__(self, results: Optional[dict[str, ProviderResult]] = None, default: Optional[ProviderResult] = None):
        self._results = results or {}
        self._default = default

    def available(self) -> bool:
        return True

    def research(self, question: str) -> ProviderResult:
        result = self._results.get(question, self._default)
        if result is None:
            raise ProviderError(f"StaticProvider has no result for: {question!r}")
        return result


def exa_search_sources(api_key: str, question: str, client: Optional[httpx.Client] = None, limit: int = 8) -> list[Source]:
    """Web search via Exa: the fresh-facts half of local research.

    The local Qwen3 model writes the answer; this is what lets it answer
    from current, real sources instead of model memory.
    """
    http = client or httpx.Client(timeout=60)
    response = http.post(
        "https://api.exa.ai/search",
        headers={"x-api-key": api_key},
        json={"query": question, "numResults": limit, "contents": {"text": True}},
    )
    if response.status_code != 200:
        raise ProviderError(f"Exa API error {response.status_code}: {response.text[:200]}")
    sources = []
    for r in response.json().get("results", []):
        sources.append(
            Source(
                title=r.get("title", ""),
                url=r.get("url", ""),
                published_at=r.get("publishedDate"),
                snippet=(r.get("text") or "")[:500],
            )
        )
    return sources


_SYNTHESIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are JARVIS, a research assistant. Answer the question using ONLY "
            "the sources below. After each sentence that uses a source, cite its "
            "number in square brackets, like [1] or [2]. If the sources do not "
            "contain the answer, say what is missing instead of guessing.\n\n"
            "Sources:\n{sources}",
        ),
        ("human", "{question}"),
    ]
)


def _format_sources(sources: list[Source]) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s.title} - {s.url}\n{s.snippet or ''}")
    return "\n\n".join(lines)


class LocalQwenResearchProvider:
    """Default research provider: self-hosted Qwen3 8B + Exa web search.

    LangChain builds the synthesis chain (prompt -> local chat model ->
    text). Web search is deliberately separate: without EXA_API_KEY this
    provider reports itself unconfigured rather than letting the model
    answer fresh questions from memory and present guesses as research.
    """

    name = "local_qwen"
    configuration_hint = (
        "Fresh research needs web search: set EXA_API_KEY (the local Qwen3 "
        "model writes the answer but cannot browse the web by itself), and "
        "start your local model server (docs/local-model.md). To use a "
        "hosted provider instead, set JARVIS_RESEARCH_PROVIDER=perplexity "
        "or openai with its API key."
    )

    def __init__(
        self,
        settings: Settings,
        model: Optional[BaseChatModel] = None,
        client: Optional[httpx.Client] = None,
    ):
        self._settings = settings
        self._model = model
        self._client = client

    def available(self) -> bool:
        return bool(self._settings.exa_api_key)

    def research(self, question: str) -> ProviderResult:
        sources = exa_search_sources(
            self._settings.exa_api_key or "", question, client=self._client
        )
        if not sources:
            # No evidence: the pipeline marks this UNCERTAIN. Never let the
            # model fill the gap from memory.
            return ProviderResult()
        model = self._model or build_chat_model(self._settings)
        chain = _SYNTHESIS_PROMPT | model | StrOutputParser()
        try:
            text = chain.invoke(
                {"question": question, "sources": _format_sources(sources)}
            )
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(
                "Could not reach the local model server at "
                f"{self._settings.local_llm_base_url} ({exc}). Is Qwen3 8B "
                "running there? See docs/local-model.md."
            ) from exc
        return ProviderResult(
            summary=text,
            claims=claims_from_cited_text(text, len(sources)),
            sources=sources,
        )


def provider_from_settings(settings: Settings) -> ResearchProvider:
    """Pick the research provider.

    Default is the self-hosted local Qwen3 provider. Hosted providers are
    opt-in only, via JARVIS_RESEARCH_PROVIDER, so no paid API is called
    unless the user explicitly chose it.
    """
    requested = (settings.research_provider or "local").strip().lower()
    if requested == "local":
        return LocalQwenResearchProvider(settings)
    if requested == "openai":
        return OpenAIWebSearchProvider(settings.openai_api_key)
    if requested == "perplexity":
        return PerplexityProvider(settings.perplexity_api_key)
    if requested == "exa":
        return ExaProvider(settings.exa_api_key)
    return UnconfiguredProvider()


class UnconfiguredProvider:
    name = "unconfigured"
    configuration_hint = (
        "Set JARVIS_RESEARCH_PROVIDER to local, openai, perplexity or exa "
        "(local needs EXA_API_KEY for web search; the hosted ones need "
        "OPENAI_API_KEY, PERPLEXITY_API_KEY or EXA_API_KEY)."
    )

    def available(self) -> bool:
        return False

    def research(self, question: str) -> ProviderResult:
        raise ProviderError(f"No research provider configured. {self.configuration_hint}")
