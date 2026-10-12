"""A shared, recoverable model-call allowance; missing usage stays unknown."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from agent_core.contracts.errors import AgentLimitReachedError


@dataclass
class PlanModelBudget:
    max_calls: int | None = 64
    calls: int = 0
    recorded_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    unknown_usage_calls: int = 0
    revision: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def __post_init__(self) -> None:
        if self.max_calls is not None and self.max_calls < 1:
            raise ValueError("max_calls must be positive or None")

    def begin_call(self) -> None:
        with self._lock:
            if self.max_calls is not None and self.calls >= self.max_calls:
                raise AgentLimitReachedError("Plan model-call allowance exhausted")
            self.calls += 1
            self.revision += 1

    def record_usage(self, message_or_none: Any) -> None:
        with self._lock:
            if self.recorded_calls >= self.calls:
                raise ValueError("Model usage was reported without a started call")
            usage = getattr(message_or_none, "usage_metadata", None)
            if isinstance(message_or_none, Mapping):
                usage = message_or_none.get("usage_metadata", message_or_none)
            valid = isinstance(usage, Mapping) and all(isinstance(usage.get(key), int) and not isinstance(usage[key], bool)
                                                     and usage[key] >= 0 for key in ("input_tokens", "output_tokens", "total_tokens"))
            if valid:
                self.input_tokens += usage["input_tokens"]
                self.output_tokens += usage["output_tokens"]
                self.total_tokens += usage["total_tokens"]
            else:
                self.unknown_usage_calls += 1
            self.recorded_calls += 1
            self.revision += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"revision": self.revision, "calls": self.calls, "recorded_calls": self.recorded_calls,
                    "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                    "total_tokens": self.total_tokens, "unknown_usage_calls": self.unknown_usage_calls}

    def restore(self, value: Mapping[str, Any] | None) -> None:
        if not value:
            return
        keys = ("revision", "calls", "recorded_calls", "input_tokens", "output_tokens", "total_tokens", "unknown_usage_calls")
        if any(not isinstance(value.get(key), int) or isinstance(value[key], bool) or value[key] < 0 for key in keys):
            raise ValueError("Invalid saved plan model allowance")
        if value["recorded_calls"] > value["calls"] or value["unknown_usage_calls"] > value["recorded_calls"]:
            raise ValueError("Inconsistent saved model usage")
        with self._lock:
            if value["revision"] < self.revision:
                return
            for key in keys:
                setattr(self, key, value[key])

    def public(self) -> dict[str, Any]:
        data = self.snapshot()
        unknown = data["unknown_usage_calls"] + data["calls"] - data["recorded_calls"]
        return {"modelCalls": data["calls"], "maxModelCalls": self.max_calls,
                "inputTokens": None if unknown else data["input_tokens"],
                "outputTokens": None if unknown else data["output_tokens"],
                "totalTokens": None if unknown else data["total_tokens"],
                "unknownUsageCalls": unknown}


__all__ = ["PlanModelBudget"]
