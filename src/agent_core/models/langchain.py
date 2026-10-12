"""Adapt the owned provider stream to LangChain's public chat-model contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from agent_core.runtime.streaming import call_async, run_sync
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

from agent_core.contracts.errors import ToolCallingUnsupportedError


class StreamingChatModelAdapter(BaseChatModel):
    """Keep native agent binding while collecting each provider call once.

    ``collect`` owns cancellation, merged usage and any candidate-text handling.
    LangChain still invokes the model and handles its response through the normal
    middleware chain. Provider clients remain owned and closed by the Run.
    """

    delegate: Any = Field(exclude=True, repr=False)
    collect: Callable[..., AIMessage] = Field(exclude=True, repr=False)
    cache: bool = False

    @property
    def _llm_type(self) -> str:
        return "agent-core-owned-stream"

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> StreamingChatModelAdapter:
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        try:
            bound = self.delegate.bind_tools(tools, **kwargs)
        except (AttributeError, NotImplementedError) as error:
            raise ToolCallingUnsupportedError("模型不支持工具调用") from error
        return self.model_copy(update={"delegate": bound})

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return run_sync(self._agenerate(messages, stop=stop, **kwargs))

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        if stop is not None:
            kwargs["stop"] = stop
        response = await call_async(self.collect, self.delegate, messages, model_kwargs=kwargs)
        return ChatResult(generations=[ChatGeneration(message=response)])
