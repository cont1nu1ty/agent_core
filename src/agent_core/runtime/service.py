"""Execute one invocation; queues, business attempts and persistence are external."""
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import AsyncExitStack
import asyncio
import inspect
from typing import Any, Literal
from uuid import uuid4

from langgraph.errors import GraphRecursionError

from agent_core.config import AgentConfig
from agent_core.contracts.context import RunContext
from agent_core.contracts.errors import (
    AgentCancelled, AgentContextError, AgentLimitReachedError, AgentRecursionLimitError,
    AgentTimeoutError, AgentUpstreamError, ModelConfigurationError,
    ObservationValidationError,
)
from agent_core.contracts.events import RunEvent
from agent_core.contracts.runs import RunRequest
from agent_core.models.streaming import astream_chat_message
from agent_core.observability import UsageObserver, NoopUsageObserver
from agent_core.observability.usage import arecord_model_call
from agent_core.runtime.graph_runner import astream_graph
from agent_core.runtime.streaming import call_async, run_sync, iter_sync
from agent_core.workflows.contracts import RunResources, PersistenceResources
from agent_core.workflows.registry import WorkflowRegistry


class _SyncPort:
    """Compatibility for synchronous host ports; never used by async execution."""
    run = staticmethod(run_sync)


class AgentService:
    def __init__(
        self, *, workflows: WorkflowRegistry,
        model_factory: Callable[[], Any] | None = None,
        settings: AgentConfig | None = None, usage_observer: UsageObserver | None = None,
        checkpointer: Any = None, store: Any = None,
        model_ownership: Literal["owned", "borrowed"] = "owned",
        persistence_factory: Callable[[], Any] | None = None,
    ) -> None:
        if model_ownership not in {"owned", "borrowed"}:
            raise ValueError("model_ownership must be owned or borrowed")
        self.workflows = workflows
        self.model_factory = model_factory
        self.settings = settings or AgentConfig()
        self.checkpointer = checkpointer
        self.store = store
        self.model_ownership = model_ownership
        self.persistence_factory = persistence_factory
        self.usage_observer = usage_observer if usage_observer is not None else NoopUsageObserver()

    def stream(self, request: RunRequest, context: RunContext) -> Iterator[RunEvent]:
        """Drive the same async generator for synchronous hosts, with backpressure."""
        yield from iter_sync(self.astream(request, context))

    async def astream(self, request: RunRequest, context: RunContext) -> AsyncIterator[RunEvent]:
        """Native async execution; all resources stay on the caller's event loop."""
        context.cancellation.raise_if_cancelled()
        definition = self.workflows.resolve(request.workflow_id)
        model = None
        loop, task = asyncio.get_running_loop(), asyncio.current_task()

        def cancel_task() -> None:
            if task is not None and not task.done():
                loop.call_soon_threadsafe(task.cancel)

        async def call_model(model: Any, messages: Any, cancellation: Any, **kwargs: Any) -> Any:
            phase = kwargs.pop("phase", "answer" if kwargs.get("on_text") else "decision")
            kwargs.pop("async_stream_runner", None)
            step_id = kwargs.pop("step_id", None)
            return await arecord_model_call(
                lambda: astream_chat_message(model, messages, cancellation, **kwargs),
                observer=self.usage_observer, run_id=request.run_id,
                attempt=request.attempt, call_id=str(uuid4()), phase=phase, step_id=step_id,
            )

        context.cancellation.bind_stream(cancel_task)
        try:
            async with AsyncExitStack() as stack:
                persistence = PersistenceResources(self.checkpointer, self.store)
                if self.persistence_factory is not None:
                    manager = self.persistence_factory()
                    persistence = await manager.__aenter__()

                    async def release_resources(error_type, error, traceback):
                        try:
                            await manager.__aexit__(error_type, error, traceback)
                        except BaseException:
                            if error is None:
                                raise
                        # A resource manager cannot suppress a failed Run.
                        return False

                    stack.push_async_exit(release_resources)
                model = self.model_factory() if self.model_factory is not None else None
                if inspect.isawaitable(model):
                    model = await model
                resources = RunResources(model, self.settings, _SyncPort(), call_model,
                    persistence.checkpointer, persistence.store, call_model)
                execution = definition.factory(request, context, resources)
                if inspect.isawaitable(execution):
                    execution = await execution
                events = astream_graph(execution, request, context, recursion_limit=self.settings.recursion_limit)
                try:
                    async for event in events:
                        yield event
                finally:
                    await events.aclose()
        except asyncio.CancelledError as error:
            if context.cancellation.cancelled:
                raise AgentCancelled("Agent run cancelled") from error
            raise
        except (AgentCancelled, AgentContextError, AgentLimitReachedError, AgentUpstreamError, ModelConfigurationError):
            raise
        except GraphRecursionError as error:
            raise AgentRecursionLimitError("Agent exceeded graph recursion limit") from error
        except ObservationValidationError as error:
            raise AgentContextError("Tool result violated its trusted boundary") from error
        except Exception as error:
            if context.cancellation.cancelled:
                raise AgentCancelled("Agent run cancelled") from error
            if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold():
                raise AgentTimeoutError("Model response timed out") from error
            raise AgentUpstreamError("Model temporarily unavailable") from error
        finally:
            context.cancellation.unbind_stream(cancel_task)
            if model is not None and self.model_ownership == "owned":
                close = getattr(model, "aclose", None) or getattr(model, "close", None)
                if callable(close):
                    try:
                        await call_async(close)
                    except (Exception, asyncio.CancelledError):
                        pass
