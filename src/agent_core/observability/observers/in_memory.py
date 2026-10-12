"""Small in-process usage observer for tests and local integrations."""
from __future__ import annotations

from threading import Lock

from agent_core.observability.contracts import UsageRecord


class InMemoryUsageObserver:
    def __init__(self) -> None:
        self._records: list[UsageRecord] = []
        self._lock = Lock()

    def record(self, usage: UsageRecord) -> None:
        with self._lock:
            self._records.append(usage)

    @property
    def records(self) -> tuple[UsageRecord, ...]:
        with self._lock:
            return tuple(self._records)
