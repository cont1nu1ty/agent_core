"""Per-invocation control and explicitly scoped, workflow-owned input."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Generic, Protocol, TypeVar
from agent_core.memory.history import HistoryReader

Extension = TypeVar("Extension")

class RunControl(Protocol):
    @property
    def cancelled(self) -> bool: ...
    def raise_if_cancelled(self) -> None: ...
    def bind_stream(self, closer: Callable | None) -> None: ...
    def unbind_stream(self, closer: Callable | None) -> None: ...

@dataclass(frozen=True)
class RunContext(Generic[Extension]):
    cancellation: RunControl
    history: HistoryReader | None = None
    extension: Extension | None = None
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc))
