"""Host-provided conversation snapshots, independent of session storage."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class HistoryMessage:
    """One completed user/assistant pair, already filtered by the host."""

    user_message: str
    assistant_answer: str
    source_run_id: str | None = None
    user_source_id: str | None = None
    assistant_source_id: str | None = None
    task_id: str | None = None


class HistoryReader(Protocol):
    def read(self) -> Sequence[HistoryMessage]:
        """Return eligible pairs in chronological order."""
        ...


class SummaryCallReservation(Protocol):
    def __call__(self, call_id: str, limit: int) -> None:
        """Atomically reserve one model attempt or raise before sending it.

        A trusted host binds the Run identity. Reservations are never refunded
        after uncertain failures; this budget is distinct from execution state.
        """
        ...


class AsyncHistoryReader(Protocol):
    async def aread(self) -> Sequence[HistoryMessage]:
        """Return the authorized history using the caller's event loop."""
        ...
