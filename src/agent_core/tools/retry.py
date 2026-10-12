"""Bounded, cancellation-aware waiting for explicitly safe tool retries."""

from __future__ import annotations

from collections.abc import Callable
import asyncio
from agent_core.runtime.streaming import call_async, run_sync
import math
import random
import time

from agent_core.contracts.errors import ToolQueryTimeoutError
from agent_core.tools.contracts import ToolRuntime


class RetryController:
    """Inject clocks and waiting in tests; never reset an operation's deadline."""

    def __init__(self, *, wait: Callable[[float], None] = time.sleep,
                 jitter: Callable[[float, float], float] = random.uniform,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self.wait = wait
        self.jitter = jitter
        self.monotonic = monotonic

    def remaining(self, runtime: ToolRuntime, deadline: float) -> float:
        runtime.raise_if_cancelled()
        remaining = deadline - self.monotonic()
        if runtime.deadline is not None:
            remaining = min(remaining, (runtime.deadline - runtime.clock()).total_seconds())
        if remaining <= 0:
            raise ToolQueryTimeoutError("tool execution deadline exceeded")
        return remaining

    async def await_before_retry(self, runtime: ToolRuntime, *, deadline: float,
                          retry_number: int, retry_after_seconds: float | None = None) -> bool:
        remaining = self.remaining(runtime, deadline)
        if retry_number not in {1, 2}:
            return False
        if retry_after_seconds is not None and (
            not math.isfinite(retry_after_seconds) or not 0 <= retry_after_seconds <= 2
        ):
            return False
        delay = (0.25 if retry_number == 1 else 0.75) * self.jitter(0.8, 1.2)
        delay = max(delay, retry_after_seconds or 0.0)
        if delay >= remaining:
            return False
        target = self.monotonic() + delay
        while True:
            self.remaining(runtime, deadline)
            rest = target - self.monotonic()
            if rest <= 0:
                return True
            if self.wait is time.sleep:
                await asyncio.sleep(min(0.05, rest))
            else:
                await call_async(self.wait, min(0.05, rest))

    def wait_before_retry(self, runtime, **kwargs):
        return run_sync(self.await_before_retry(runtime, **kwargs))
