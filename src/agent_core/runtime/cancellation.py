"""Cooperative task cancellation and active-stream interruption."""
from __future__ import annotations

from collections.abc import Callable
from threading import Event, Lock
from typing import Any

from agent_core.contracts.errors import AgentCancelled


class CancellationToken:
    """Cancellation token that can also close an active provider stream."""

    def __init__(self) -> None:
        self._event = Event()
        self._lock = Lock()
        self._closers: list[Callable[[], Any]] = []

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise AgentCancelled("Agent任务已取消")

    def bind_stream(self, closer: Callable[[], Any] | None) -> None:
        if closer is None:
            return
        close_now = False
        with self._lock:
            if self._event.is_set():
                close_now = True
            else:
                self._closers.append(closer)
        if close_now:
            try:
                closer()
            except Exception:
                pass

    def unbind_stream(self, closer: Callable[[], Any] | None) -> None:
        if closer is None:
            return
        with self._lock:
            self._closers = [active for active in self._closers if active is not closer]

    def cancel(self) -> None:
        closer: Callable[[], Any] | None
        with self._lock:
            self._event.set()
            # Nested provider streams temporarily override the Run callback.
            # Removing the provider restores cancellation of tools/checkpoints.
            closer = self._closers[-1] if self._closers else None
            self._closers.clear()
        if closer is not None:
            try:
                closer()
            except Exception:
                # Closing a stream while it reaches natural EOF is harmless.
                pass
