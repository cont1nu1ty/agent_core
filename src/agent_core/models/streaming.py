"""Collect one provider message while keeping its stream cancellable."""
from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, BaseMessageChunk
from langchain_core.messages.utils import message_chunk_to_message

from agent_core.contracts.errors import AgentCancelled, AgentUpstreamError, ToolCallingUnsupportedError
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.streaming import AsyncStreamRunner, call_async


def _accumulate_model_item(
    item: Any,
    combined: BaseMessageChunk | None,
    complete: AIMessage | None,
    on_text: Callable[[str], None] | None,
) -> tuple[BaseMessageChunk | None, AIMessage | None]:
    if isinstance(item, BaseMessageChunk):
        if on_text is not None:
            visible = str(item.text)
            if visible:
                on_text(visible)
        return item if combined is None else combined + item, complete
    if isinstance(item, AIMessage):
        if on_text is not None:
            visible = str(item.text)
            if visible:
                on_text(visible)
        return combined, item
    raise ToolCallingUnsupportedError(
        "模型返回了不受支持的消息类型"
    )


async def _collect_async_chat_message(
    model: Any,
    messages: Sequence[BaseMessage],
    cancellation: CancellationToken,
    on_text: Callable[[str], None] | None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> AIMessage:
    """Consume an async provider stream that can be cancelled cross-thread."""
    provider_stream = model.astream(messages, **(model_kwargs or {}))
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    if task is None:
        raise AgentUpstreamError("模型异步流缺少运行任务")

    def cancel_task() -> None:
        if not task.done():
            loop.call_soon_threadsafe(task.cancel)

    cancellation.bind_stream(cancel_task)
    combined: BaseMessageChunk | None = None
    complete: AIMessage | None = None
    try:
        async for item in provider_stream:
            cancellation.raise_if_cancelled()
            combined, complete = _accumulate_model_item(
                item,
                combined,
                complete,
                on_text,
            )
    except asyncio.CancelledError as error:
        if cancellation.cancelled:
            raise AgentCancelled("Agent任务已取消") from error
        if not task.cancelling():
            raise AgentUpstreamError("模型流意外中止") from error
        raise
    finally:
        cancellation.unbind_stream(cancel_task)
        close = getattr(provider_stream, "aclose", None)
        if callable(close):
            try:
                await close()
            except asyncio.CancelledError as error:
                if not cancellation.cancelled:
                    raise AgentUpstreamError(
                        "模型流关闭异常"
                    ) from error
            except Exception:
                pass
    cancellation.raise_if_cancelled()
    if combined is not None:
        message = message_chunk_to_message(combined)
        if isinstance(message, AIMessage):
            return message
    if complete is not None:
        return complete
    raise AgentUpstreamError("模型返回空结果")


def stream_chat_message(
    model: Any,
    messages: Sequence[BaseMessage],
    cancellation: CancellationToken,
    *,
    async_stream_runner: AsyncStreamRunner,
    on_text: Callable[[str], None] | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> AIMessage:
    """Collect one model message while keeping the active stream cancellable."""
    cancellation.raise_if_cancelled()
    if callable(getattr(model, "astream", None)):
        return async_stream_runner.run(
            _collect_async_chat_message(
                model,
                messages,
                cancellation,
                on_text,
                model_kwargs,
            )
        )
    provider_stream = model.stream(messages, **(model_kwargs or {}))
    close = getattr(provider_stream, "close", None)
    closer = close if callable(close) else None
    cancellation.bind_stream(closer)
    combined: BaseMessageChunk | None = None
    complete: AIMessage | None = None
    try:
        for item in provider_stream:
            cancellation.raise_if_cancelled()
            combined, complete = _accumulate_model_item(
                item,
                combined,
                complete,
                on_text,
            )
    finally:
        cancellation.unbind_stream(closer)
        if closer is not None:
            try:
                closer()
            except Exception:
                pass
    cancellation.raise_if_cancelled()
    if combined is not None:
        message = message_chunk_to_message(combined)
        if isinstance(message, AIMessage):
            return message
    if complete is not None:
        return complete
    raise AgentUpstreamError("模型返回空结果")


async def astream_chat_message(model, messages, cancellation, *, on_text=None,
                              model_kwargs=None, async_stream_runner=None):
    """Collect on the caller loop; only legacy sync providers use a worker."""
    cancellation.raise_if_cancelled()
    if callable(getattr(model, "astream", None)):
        return await _collect_async_chat_message(model, messages, cancellation, on_text, model_kwargs)
    return await call_async(stream_chat_message, model, messages, cancellation,
                            async_stream_runner=None, on_text=on_text, model_kwargs=model_kwargs)
