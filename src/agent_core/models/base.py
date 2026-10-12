"""Explicit provider profile and owned chat-model resources."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ChatModel:
    """ChatOpenAI plus the per-run HTTP clients owned by this adapter."""

    chat_model: Any
    _sync_http_client: Any = field(repr=False)
    _async_http_client: Any = field(repr=False)
    max_tokens: int | None = None
    supports_required_tool_choice: bool = True

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        if not self.supports_required_tool_choice and kwargs.get("tool_choice") in ("any", "required"):
            # Some compatible gateways only support automatic tool selection.
            # Structured-response validation still belongs to create_agent.
            kwargs["tool_choice"] = "auto"
        return self.chat_model.bind_tools(tools, **kwargs)

    def stream(self, messages: Any, **kwargs: Any) -> Any:
        return self.chat_model.stream(messages, **kwargs)

    def astream(self, messages: Any, **kwargs: Any) -> Any:
        return self.chat_model.astream(messages, **kwargs)

    async def aclose(self) -> None:
        from agent_core.runtime.streaming import call_async
        try:
            await self._async_http_client.aclose()
        finally:
            await call_async(self._sync_http_client.close)

    def close(self) -> None:
        self._sync_http_client.close()
