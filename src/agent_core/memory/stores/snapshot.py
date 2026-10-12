"""Immutable history supplied by the host for one invocation."""
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from agent_core.memory.history import HistoryMessage


@dataclass(frozen=True, init=False)
class SnapshotHistory:
    """An immutable per-run snapshot; this is not a persistence backend."""

    messages: tuple[HistoryMessage, ...]
    preselected: bool
    context_summary: dict[str, Any] | None
    warning: str | None
    context_history_dependencies: tuple[str, ...]
    context_has_legacy_history: bool
    archived: bool

    def __init__(self, messages: Sequence[HistoryMessage] = (), *, preselected: bool = False,
                 context_summary: dict[str, Any] | None = None, warning: str | None = None,
                 context_history_dependencies: Sequence[str] = (), context_has_legacy_history: bool = False,
                 archived: bool = False) -> None:
        object.__setattr__(self, "messages", tuple(messages))
        object.__setattr__(self, "preselected", preselected)
        object.__setattr__(self, "context_summary", context_summary)
        object.__setattr__(self, "warning", warning)
        object.__setattr__(self, "context_history_dependencies", tuple(context_history_dependencies))
        object.__setattr__(self, "context_has_legacy_history", context_has_legacy_history)
        object.__setattr__(self, "archived", archived)

    def read(self) -> tuple[HistoryMessage, ...]:
        return self.messages
