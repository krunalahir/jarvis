"""Content creation: turn a topic into a platform-ready post draft.

Uses the configured chat model through LangChain: the self-hosted Qwen3
8B by default, hosted OpenAI only with JARVIS_LLM_PROVIDER=openai. If the
model cannot be reached, generation is reported unavailable - JARVIS does
not pretend to write content it cannot. Generating a draft never
publishes anything; publishing is a separate, approved action.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .config import Settings
from .llm import LLMNotConfigured, build_chat_model
from .schemas import Platform


class ContentGenerationUnavailable(Exception):
    pass


_LENGTH_HINTS = {
    Platform.X: "at most 280 characters",
    Platform.INSTAGRAM: "a caption of a few short paragraphs",
    Platform.YOUTUBE: "a title (max 100 chars) on the first line, then a description",
    Platform.SNAPCHAT: "a short caption",
}

_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are JARVIS, a writing assistant. Write {length} for "
            "{platform} about the user's topic. Tone: {tone}. "
            "Return only the post text.",
        ),
        ("human", "{topic}"),
    ]
)


class ContentGenerator:
    def __init__(self, settings: Settings, model: Optional[BaseChatModel] = None):
        self._settings = settings
        self._model = model

    def available(self) -> bool:
        # "local" needs no key; reachability is checked when generating.
        # "openai" is a paid opt-in and needs its key up front.
        if self._settings.llm_provider == "openai":
            return bool(self._settings.openai_api_key)
        return True

    def generate(self, platform: Platform, topic: str, tone: Optional[str] = None) -> str:
        if not self.available():
            raise ContentGenerationUnavailable(
                "JARVIS_LLM_PROVIDER=openai needs OPENAI_API_KEY. Leave the "
                "provider as 'local' to use your self-hosted Qwen3 8B, or "
                "write the post text yourself and create the draft directly."
            )
        try:
            model = self._model or build_chat_model(self._settings)
        except LLMNotConfigured as exc:
            raise ContentGenerationUnavailable(str(exc)) from exc
        chain = _PROMPT | model | StrOutputParser()
        try:
            text = chain.invoke(
                {
                    "platform": platform.value,
                    "topic": topic,
                    "tone": tone or "clear and direct",
                    "length": _LENGTH_HINTS.get(platform, "a short post"),
                }
            )
        except Exception as exc:
            raise ContentGenerationUnavailable(
                "Could not reach the local model server at "
                f"{self._settings.local_llm_base_url} ({exc}). Is Qwen3 8B "
                "running there? See docs/local-model.md."
            ) from exc
        return text.strip()
