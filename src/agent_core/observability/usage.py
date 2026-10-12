"""Best-effort usage reporting with exactly one record per model call."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from agent_core.contracts.errors import AgentCancelled
from agent_core.models.usage import normalize_usage
from agent_core.runtime.streaming import call_async, run_sync
from agent_core.observability.contracts import UsageObserver, UsageRecord


MessageT = TypeVar("MessageT")


async def arecord_model_call(
    operation: Callable[[], MessageT],
    *,
    observer: UsageObserver | None,
    run_id: str,
    attempt: int,
    call_id: str,
    phase: str,
    step_id: str | None = None,
) -> MessageT:
    """Observe a completed, merged message, never accumulate stream chunks.

    Failed or cancelled streams have unknown usage. Observers receive no
    messages, prompts, tool payloads, credentials, or provider exceptions.
    An observer failure cannot change the task result or its original error.
    """
    status: Literal["succeeded", "failed", "cancelled"] = "failed"
    result: Any = None
    try:
        result = await call_async(operation)
        status = "succeeded"
        return result
    except (AgentCancelled, asyncio.CancelledError):
        status = "cancelled"
        raise
    finally:
        if observer is not None:
            try:
                counts = normalize_usage(result) if status == "succeeded" else {}
                await call_async(observer.record,
                    UsageRecord(
                        run_id=run_id,
                        attempt=attempt,
                        call_id=call_id,
                        phase=phase,
                        step_id=step_id,
                        status=status,
                        **counts,
                    )
                )
            except Exception:
                # Telemetry must never become a dependency of agent execution.
                pass


def record_model_call(operation, **kwargs):
    return run_sync(arecord_model_call(operation, **kwargs))
