"""Tool contracts independent of application resources and result schemas."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Awaitable, Literal, Protocol

from pydantic import BaseModel

from agent_core.contracts.artifacts import Evidence
from agent_core.runtime.budget import ExecutionBudget


class ToolRuntime(Protocol):
    execution: ExecutionBudget
    clock: Callable[[], datetime]
    run_awaitable: Callable[[Awaitable[Any]], Any] | None

    @property
    def deadline(self) -> datetime | None: ...

    def with_timeout(self, seconds: float) -> ToolRuntime: ...

    def raise_if_cancelled(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    content: str
    succeeded: bool
    reference: Evidence | None = None
    message: str = ""
    data: Any = None
    sensitive_values: tuple[str, ...] = ()

    @property
    def content_size(self) -> int:
        return len(self.content.encode("utf-8"))


@dataclass(frozen=True, slots=True)
class RecoverableToolError:
    code: str
    message: str
    retryable: bool = False
    failure_kind: str = "recoverable"
    retry_after_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class ToolRetryPolicy:
    """Trusted replay permission; error text never grants this permission."""

    mode: Literal["never", "safe_read"] = "never"
    max_extra_retries: int = 2

    def __post_init__(self) -> None:
        if self.mode not in {"never", "safe_read"}:
            raise ValueError("unknown tool retry mode")
        if type(self.max_extra_retries) is not int or not 0 <= self.max_extra_retries <= 2:
            raise ValueError("max_extra_retries must be between zero and two")


@dataclass(frozen=True, slots=True)
class ToolDiscoverySpec:
    """Model-facing discovery metadata supplied by the trusted tool owner."""

    exposure: Literal["direct", "deferred", "hidden"] = "deferred"
    direct_priority: int = 0
    required: bool = False
    aliases: tuple[str, ...] = ()
    when_to_use: str = ""
    when_not_to_use: str = ""
    intent_clusters: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.exposure not in {"direct", "deferred", "hidden"}:
            raise ValueError("unknown tool exposure")
        if self.required and self.exposure != "direct":
            raise ValueError("required tools must have direct exposure")
        if any(not isinstance(alias, str) or not alias.strip() for alias in self.aliases):
            raise ValueError("tool aliases must be nonempty text")
        if (not isinstance(self.intent_clusters, (tuple, list))
                or any(not isinstance(cluster, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", cluster)
                       for cluster in self.intent_clusters)
                or len(set(self.intent_clusters)) != len(self.intent_clusters)):
            raise ValueError("intent clusters must be unique snake_case identifiers")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "intent_clusters", tuple(self.intent_clusters))


@dataclass(frozen=True, slots=True)
class AuthorizationDecision:
    allowed: bool
    code: str = "allowed"
    message: str = ""


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    handler: Callable[[BaseModel, ToolRuntime], Any | Awaitable[Any]]
    version: str = "1.0.0"
    category: str = "general"
    max_calls_per_run: int = 8
    timeout_seconds: float = 10.0
    tags: frozenset[str] = frozenset()
    result_builder: Callable[[BaseModel, Any, ToolRuntime], ToolResult] | None = None
    recoverable_error_mapper: Callable[[Exception], RecoverableToolError | None] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    discovery: ToolDiscoverySpec = field(default_factory=ToolDiscoverySpec)
    retry_policy: ToolRetryPolicy = field(default_factory=ToolRetryPolicy)
    source: str = "application"
    budget_group: str = "external"
    effect: Literal["read", "write", "control", "unknown"] = "unknown"

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]{1,63}", self.name):
            raise ValueError(f"invalid tool name: {self.name!r}")
        if not re.fullmatch(r"[1-9][0-9]*\.[0-9]+\.[0-9]+", self.version):
            raise ValueError("tool version must be semantic major.minor.patch")
        if not self.category.strip() or not self.description.strip():
            raise ValueError("tool category and description cannot be blank")
        if not isinstance(self.input_schema, type) or not issubclass(self.input_schema, BaseModel):
            raise TypeError("input_schema must be a Pydantic model")
        if self.max_calls_per_run < 1:
            raise ValueError("max_calls_per_run must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not callable(self.handler):
            raise TypeError("handler must be callable")
        if not isinstance(self.discovery, ToolDiscoverySpec) or not isinstance(self.retry_policy, ToolRetryPolicy):
            raise TypeError("tool discovery and retry policies must be typed")
        if self.budget_group not in {"external", "memory", "discovery"}:
            raise ValueError("unknown tool budget group")
        if self.effect not in {"read", "write", "control", "unknown"}:
            raise ValueError("unknown tool effect")
        if self.retry_policy.mode == "safe_read" and self.effect != "read":
            raise ValueError("automatic retries require an explicitly read-only tool")
        for name in ("result_builder", "recoverable_error_mapper"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"{name} must be callable")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def model_json_schema(self) -> dict[str, Any]:
        return self.input_schema.model_json_schema()


class ToolAuthorizer(Protocol):
    def authorize(self, definition: ToolSpec, runtime: ToolRuntime) -> AuthorizationDecision | Awaitable[AuthorizationDecision]: ...


class NativeAuthorizer(Protocol):
    """Host policy for a registered native tool and its concrete arguments."""

    def authorize(self, *, tool_name: str, arguments: Mapping[str, Any],
                  runtime: Any, metadata: Mapping[str, Any]) -> AuthorizationDecision | Awaitable[AuthorizationDecision]: ...
