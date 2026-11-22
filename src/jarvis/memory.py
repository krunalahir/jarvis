"""Conversation memory for the agent.

Each conversation is remembered on the server in three parts:

- summary: the gist of the older conversation. Rewritten by the model,
  but only when the recent turns pass the limit, never on every message.
- turns:   the most recent messages, word for word (3 to 15 by default).
- facts:   exact details the user stated - email addresses, phone numbers,
  dates, and sentences such as "my name is ..." or "remember that ...".
  These are copied by plain text matching, never rewritten by the model,
  so an address or a date cannot be changed by a bad summary.

The client then sends only the new message plus a conversation id.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel

from .config import Settings
from .llm import build_chat_model
from .research.store import Store
from .schemas import ConversationMemory, utcnow

log = logging.getLogger(__name__)

MAX_FACTS = 40
MAX_FACT_LENGTH = 200
CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_MONTH = r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.?"
_DATE = re.compile(
    r"\b(?:\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    rf"|\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTH}(?:,?\s+\d{{4}})?"
    rf"|{_MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,?\s+\d{{4}})?)\b",
    re.IGNORECASE,
)
_PHONE = re.compile(r"(?<![\w@.])\+?\d[\d\s-]{8,14}\d(?![\w@])")
_STATED = re.compile(
    r"\b(?:my name is|i am called|call me|(?:his|her|their) name is|"
    r"name is|remember that|please remember|remember:)",
    re.IGNORECASE,
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

_SUMMARY_INSTRUCTIONS = (
    "You keep the running summary of a conversation between a user and "
    "their assistant JARVIS. Write the updated summary in at most 120 "
    "words. Keep what the user asked for, what was done or queued, and "
    "what is still open. Do not invent anything. Exact email addresses, "
    "phone numbers and dates are stored separately, so do not list them. "
    "Return only the summary text."
)


def extract_facts(text: str) -> list[str]:
    """Exact details stated in one user message, copied character for
    character. Plain matching on purpose: no model is involved."""
    facts: list[str] = []
    for email in _EMAIL.findall(text):
        facts.append(f"email: {email}")
    without_emails = _EMAIL.sub(" ", text)
    for date in _DATE.findall(without_emails):
        facts.append(f"date mentioned: {date.strip()}")
    without_dates = _DATE.sub(" ", without_emails)
    for phone in _PHONE.findall(without_dates):
        if sum(ch.isdigit() for ch in phone) >= 10:
            facts.append(f"phone: {phone.strip()}")
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if sentence and _STATED.search(sentence):
            facts.append(f'user said: "{sentence[:MAX_FACT_LENGTH]}"')
    return facts


class MemoryManager:
    def __init__(
        self,
        store: Store,
        settings: Settings,
        model: Optional[BaseChatModel] = None,
    ):
        self.store = store
        self._settings = settings
        self._model = model
        self.max_recent = max(2, settings.memory_max_recent)
        self.min_recent = max(1, min(settings.memory_min_recent, self.max_recent - 1))
        # If summarising keeps failing, never let the turns grow forever.
        self.hard_cap = self.max_recent * 3

    def load(self, conversation_id: str) -> ConversationMemory:
        if not CONVERSATION_ID.match(conversation_id):
            raise ValueError(
                "conversation_id may only contain letters, digits, '-' and '_' "
                "(at most 64 characters)"
            )
        return self.store.get_conversation(conversation_id) or ConversationMemory(
            id=conversation_id
        )

    def context_messages(self, memory: ConversationMemory) -> list[dict]:
        """What the agent is given before the user's new message."""
        notes = []
        if memory.facts:
            notes.append(
                "Exact details the user stated earlier. Copy them character "
                "for character when you use them:\n"
                + "\n".join(f"- {fact}" for fact in memory.facts)
            )
        if memory.summary:
            notes.append("Summary of the earlier conversation:\n" + memory.summary)
        messages: list[dict] = []
        if notes:
            messages.append(
                {
                    "role": "system",
                    "content": "Memory of this conversation (kept by JARVIS, "
                    "not typed by the user).\n\n" + "\n\n".join(notes),
                }
            )
        return messages + [dict(turn) for turn in memory.turns]

    def record(self, memory: ConversationMemory, user_message: str, reply: str) -> None:
        """Save one finished exchange and tidy the memory."""
        memory.turns.append({"role": "user", "content": user_message})
        memory.turns.append({"role": "assistant", "content": reply})
        known = {fact.lower() for fact in memory.facts}
        for fact in extract_facts(user_message):
            if fact.lower() not in known:
                memory.facts.append(fact)
                known.add(fact.lower())
        memory.facts = memory.facts[-MAX_FACTS:]
        self._compact(memory)
        memory.updated_at = utcnow()
        self.store.save_conversation(memory)

    def _compact(self, memory: ConversationMemory) -> None:
        """Past the limit, fold the oldest turns into the summary and keep
        only the newest few word for word."""
        if len(memory.turns) <= self.max_recent:
            return
        keep = self.min_recent
        # Start the kept part on a user message so it reads as a dialogue.
        while keep < len(memory.turns) - 1 and memory.turns[-keep]["role"] != "user":
            keep += 1
        older, recent = memory.turns[:-keep], memory.turns[-keep:]
        try:
            memory.summary = self._summarise(memory.summary, older)
        except Exception as exc:  # model down: keep the turns, try next time
            log.warning("could not summarise conversation %s: %s", memory.id, exc)
            memory.turns = memory.turns[-self.hard_cap:]
            return
        memory.turns = recent

    def _summarise(self, summary: str, turns: list[dict]) -> str:
        model = self._model or build_chat_model(self._settings)
        transcript = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
        result = model.invoke(
            [
                ("system", _SUMMARY_INSTRUCTIONS),
                (
                    "human",
                    f"Summary so far:\n{summary or '(nothing yet)'}\n\n"
                    f"New messages to fold in:\n{transcript}",
                ),
            ]
        )
        content = getattr(result, "content", result)
        text = _THINK.sub("", content if isinstance(content, str) else str(content)).strip()
        if not text:
            raise ValueError("the model returned an empty summary")
        return text
