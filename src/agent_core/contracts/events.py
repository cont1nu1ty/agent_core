"""Transport-independent execution events; persistence belongs to the caller."""
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunEvent:
    run_id: str
    attempt: int
    sequence: int
    kind: str
    payload: dict[str, Any] = field(default_factory=dict)
    step_id: str | None = None
    call_id: str | None = None
    activity_scope_id: str | None = None
    activity_label: str | None = None

    def as_update(self) -> dict[str, Any]:
        identity = {key: value for key, value in (
            ("stepId", self.step_id), ("callId", self.call_id),
            ("activityScopeId", self.activity_scope_id), ("activityLabel", self.activity_label),
        ) if value is not None}
        return {"kind": self.kind, **self.payload, **identity}


def activity_scope(state: Mapping[str, Any] | None) -> str:
    """Public workflow invocation identity, never a checkpoint namespace."""
    return str((state or {}).get("activity_scope_id") or "run")


def activity_label(state: Mapping[str, Any] | None) -> str:
    return str((state or {}).get("activity_label") or "任务执行")
