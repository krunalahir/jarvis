"""Shared test fakes. Nothing here touches the network."""

from __future__ import annotations

from typing import Any, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedChatModel(BaseChatModel):
    """A chat model that plays back a fixed script.

    Each entry is either an AIMessage to return or an Exception to raise.
    Tool call ids are made unique per call, the way real serving stacks
    behave, so multi-round loops build valid message histories.
    """

    script: list[Any]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> ChatResult:
        index = min(self.calls, len(self.script) - 1)
        self.calls += 1
        item = self.script[index]
        if isinstance(item, Exception):
            raise item
        msg = item
        if isinstance(msg, AIMessage) and msg.tool_calls:
            tool_calls = [
                ToolCall(name=tc["name"], args=tc["args"], id=f"{tc['id']}-{self.calls}")
                for tc in msg.tool_calls
            ]
            msg = AIMessage(content=msg.content, tool_calls=tool_calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])
