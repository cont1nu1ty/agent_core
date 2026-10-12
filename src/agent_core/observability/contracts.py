"""Stable usage records and the best-effort observer port."""
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class UsageRecord:
    run_id: str
    attempt: int
    call_id: str
    phase: str
    status: Literal["succeeded", "failed", "cancelled"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    source: Literal["reported", "unknown"] = "unknown"
    step_id: str | None = None


class UsageObserver(Protocol):
    def record(self, usage: UsageRecord) -> None: ...
