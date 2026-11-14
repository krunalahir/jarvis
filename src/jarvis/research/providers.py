"""Research providers.

A provider answers a question and returns a summary plus sources, with
claims linked to the sources that support them. Three real providers are
implemented (OpenAI, Perplexity, Exa); any one API key makes research
live. `StaticProvider` exists for tests and demos.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx

from ..config import Settings
from ..schemas import Claim, Source


@dataclass
class ProviderResult:
    summary: str = ""
    claims: list[Claim] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)


class ResearchProvider(Protocol):
    name: str

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


def provider_from_settings(settings: Settings) -> ResearchProvider:
    """Pick the first configured real provider, in a fixed order."""
    for provider in (
        OpenAIWebSearchProvider(settings.openai_api_key),
        PerplexityProvider(settings.perplexity_api_key),
        ExaProvider(settings.exa_api_key),
    ):
        if provider.available():
            return provider
    return UnconfiguredProvider()


class UnconfiguredProvider:
    name = "unconfigured"

    def available(self) -> bool:
        return False

    def research(self, question: str) -> ProviderResult:
        raise ProviderError(
            "No research provider configured. Set OPENAI_API_KEY, "
            "PERPLEXITY_API_KEY or EXA_API_KEY."
        )
