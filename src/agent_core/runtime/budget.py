"""Per-run tool limits, result accounting, and optional evidence allocation."""

from dataclasses import dataclass, field
import hashlib
import json
from threading import Lock
from typing import Any, Mapping, Sequence

from agent_core.contracts.artifacts import Evidence
from agent_core.contracts.errors import AgentContextError, AgentLimitReachedError, DuplicateToolCallError


_LIMIT_FIELDS = ("max_tool_calls", "max_external_tool_calls", "max_memory_tool_calls",
                 "max_discovery_tool_calls", "max_retry_attempts", "max_extra_retries",
                 "max_observation_bytes", "max_references")


@dataclass(slots=True)
class ExecutionBudget:
    """Per-run deterministic limits and citation allocation.

    Every rejected or failed call consumes an attempt, while strict validation
    and policy checks still happen before Service I/O. A reference number is only
    committed with a safe observation, which guarantees that failures cannot
    leave gaps in the ``R1..Rn`` sequence.
    """

    max_tool_calls: int = 8
    max_observation_bytes: int | None = 96 * 1024
    max_references: int = 64
    max_external_tool_calls: int = 8
    max_memory_tool_calls: int = 16
    max_discovery_tool_calls: int = 3
    max_retry_attempts: int = 8
    max_extra_retries: int = 2
    retries_enabled: bool = False
    discovery_enabled: bool = False
    model_requests: int = 0
    retry_reserved_count: int = 0
    call_reservations: dict[str, dict[str, Any]] = field(default_factory=dict)
    tool_call_count: int = 0
    observation_bytes: int = 0
    references: list[Evidence] = field(default_factory=list)
    calls_by_tool: dict[str, int] = field(default_factory=dict)
    calls_by_category: dict[str, int] = field(default_factory=dict)
    used_call_ids: set[str] = field(default_factory=set)
    completed_call_ids: set[str] = field(default_factory=set)
    sensitive_values: set[str] = field(default_factory=set)
    revoked_tool_names: set[str] = field(default_factory=set)
    stop_reason: str | None = None
    limit_message: str | None = None
    tool_limits: dict[str, int] = field(default_factory=dict)
    revision: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)
    _dispatchable_call_ids: set[str] = field(default_factory=set, repr=False)
    _claimed_call_ids: set[str] = field(default_factory=set, repr=False)
    _ordinary_handler_attempts: dict[str, int] = field(default_factory=dict, repr=False)
    _nonretry_dispatched_ids: set[str] = field(default_factory=set, repr=False)

    def __post_init__(self) -> None:
        if self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive")
        if self.max_observation_bytes is not None and self.max_observation_bytes < 1:
            raise ValueError("max_observation_bytes must be positive")
        if self.max_references < 1:
            raise ValueError("max_references must be positive")
        if self.max_discovery_tool_calls < 0 or self.max_retry_attempts < 0:
            raise ValueError("discovery and retry limits cannot be negative")
        if type(self.max_extra_retries) is not int or not 0 <= self.max_extra_retries <= 2:
            raise ValueError("max_extra_retries must be between zero and two")

    @property
    def logical_tool_calls(self) -> int:
        return self.tool_call_count

    def revoke_tool(self, name: str) -> None:
        """Keep a trusted authorization denial effective for the rest of this Run."""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("revoked tool name must be nonempty")
        with self._lock:
            if name not in self.revoked_tool_names:
                self.revoked_tool_names.add(name)
                self.revision += 1

    def is_tool_revoked(self, name: str) -> bool:
        with self._lock:
            return name in self.revoked_tool_names

    @staticmethod
    def _group(category: str, budget_group: str | None) -> str:
        if budget_group is None:
            return "memory" if category == "memory" else "external"
        if budget_group not in {"external", "memory", "discovery"}:
            raise ValueError("unknown tool budget group")
        return budget_group

    def _group_limit(self, group: str) -> int:
        return {"external": self.max_external_tool_calls,
                "memory": self.max_memory_tool_calls,
                "discovery": self.max_discovery_tool_calls if self.discovery_enabled else 0}[group]

    def _available(self, name: str, maximum: int | None, group: str) -> bool:
        return (self.stop_reason not in {"tool_call_limit", "observation_limit", "reference_limit"}
                and self.tool_call_count < self.max_tool_calls
                and self.calls_by_category.get(group, 0) < self._group_limit(group)
                and (maximum is None or self.calls_by_tool.get(name, 0) < maximum))

    def is_available(self, name: str, max_calls: int | None, budget_group: str = "external",
                     *, call_id: str | None = None) -> bool:
        group = self._group("external", budget_group)
        with self._lock:
            if name in self.revoked_tool_names:
                return False
            reservation = self.call_reservations.get(call_id or "")
            if (reservation is not None and reservation["tool_name"] == name
                    and reservation["budget_group"] == group and reservation["outcome"] == "pending"):
                return self.stop_reason not in {"observation_limit", "reference_limit"}
            return self._available(name, max_calls, group)

    @staticmethod
    def _argument_hash(arguments: Mapping[str, Any] | None) -> str | None:
        if arguments is None:
            return None
        # Nonfinite numbers may be rejected by the input schema; hashing them
        # must not replace the executor's safe validation feedback with an error.
        encoded = json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @staticmethod
    def _validate_call_id(call_id: str) -> str:
        call_id = call_id.strip()
        if not call_id or len(call_id) > 256:
            raise ValueError("call_id must be between 1 and 256 characters")
        return call_id

    def _charge_call(self, call_id: str, tool_name: str, maximum: int | None, group: str) -> None:
        if call_id in self.used_call_ids:
            raise DuplicateToolCallError()
        if self.stop_reason in {"tool_call_limit", "observation_limit", "reference_limit"}:
            raise AgentLimitReachedError(self.limit_message)
        if self.tool_call_count >= self.max_tool_calls:
            self.stop_reason = "tool_call_limit"
            self.limit_message = f"工具调用总额度已用完（{self.tool_call_count}/{self.max_tool_calls}）。"
            self.revision += 1
            raise AgentLimitReachedError(self.limit_message)
        # An accepted model call consumes global logical capacity even when
        # its particular tool/group has no remaining execution permission.
        self.used_call_ids.add(call_id)
        self.tool_call_count += 1
        self.revision += 1
        # A single tool or group becoming unavailable must not stop other tools.
        if self.calls_by_category.get(group, 0) >= self._group_limit(group):
            label = {"external": "查询工具", "memory": "记忆工具", "discovery": "工具发现"}[group]
            self.limit_message = f"{label}调用额度已用完（{self.calls_by_category.get(group, 0)}/{self._group_limit(group)}）。"
            raise AgentLimitReachedError(self.limit_message)
        if maximum is not None and self.calls_by_tool.get(tool_name, 0) >= maximum:
            self.limit_message = f"工具 {tool_name} 的调用额度已用完（{self.calls_by_tool.get(tool_name, 0)}/{maximum}），模型调用额度是另一项限制。"
            raise AgentLimitReachedError(self.limit_message)
        self.calls_by_tool[tool_name] = self.calls_by_tool.get(tool_name, 0) + 1
        self.calls_by_category[group] = self.calls_by_category.get(group, 0) + 1

    def reserve_call(self, *, call_id: str, tool_name: str, max_calls_for_tool: int | None,
                     budget_group: str = "external", max_extra_retries: int = 0,
                     arguments: Mapping[str, Any] | None = None, tool_version: str | None = None,
                     retry_protocol: bool = True) -> int:
        """Reserve logical work and its full retry allowance before checkpointing.

        Allowances are never refunded. A restored pending retry reservation
        must settle as unknown. Ordinary reservations preserve batch order and
        the existing unfinished-tool replay behavior, without automatic retry.
        """
        call_id = self._validate_call_id(call_id)
        group = self._group("external", budget_group)
        if type(max_extra_retries) is not int or not 0 <= max_extra_retries <= 2:
            raise ValueError("a tool may reserve at most two extra attempts")
        if type(retry_protocol) is not bool or not retry_protocol and max_extra_retries:
            raise ValueError("ordinary reservations cannot reserve retries")
        argument_hash = self._argument_hash(arguments)
        with self._lock:
            self._charge_call(call_id, tool_name, max_calls_for_tool, group)
            allowance = min(max_extra_retries, self.max_extra_retries,
                            max(0, self.max_retry_attempts - self.retry_reserved_count)) if self.retries_enabled and retry_protocol else 0
            self.retry_reserved_count += allowance
            self.call_reservations[call_id] = {
                "tool_name": tool_name, "tool_version": tool_version,
                "argument_hash": argument_hash, "budget_group": group,
                "retry_protocol": retry_protocol,
                "retry_allowance": allowance, "handler_attempts": 0, "outcome": "pending",
            }
            self._dispatchable_call_ids.add(call_id)
            return allowance

    def retry_allowance(self, call_id: str) -> int:
        with self._lock:
            return int(self.call_reservations.get(call_id, {}).get("retry_allowance", 0))

    def record_handler_attempt(self, call_id: str) -> int:
        with self._lock:
            reservation = self.call_reservations.get(call_id)
            if reservation is None:
                if call_id not in self.used_call_ids:
                    raise AgentContextError("Tool handler requires a counted logical call")
                self._ordinary_handler_attempts[call_id] = self._ordinary_handler_attempts.get(call_id, 0) + 1
                self.revision += 1
                return self._ordinary_handler_attempts[call_id]
            if call_id not in self._claimed_call_ids or reservation["outcome"] != "pending":
                raise AgentContextError("Tool retry reservation was not claimed")
            attempts = int(reservation["handler_attempts"]) + 1
            if reservation.get("retry_protocol", True):
                if attempts > 1 + int(reservation["retry_allowance"]):
                    raise AgentLimitReachedError()
            elif call_id in self._nonretry_dispatched_ids:
                raise AgentLimitReachedError()
            else:
                self._nonretry_dispatched_ids.add(call_id)
            reservation["handler_attempts"] = attempts
            self.revision += 1
            return attempts

    def pending_call_ids(self, *, retry_protocol_only: bool = True) -> tuple[str, ...]:
        with self._lock:
            return tuple(key for key, value in self.call_reservations.items()
                         if value["outcome"] == "pending" and (not retry_protocol_only or value.get("retry_protocol", True)))

    def mark_outcome_unknown(self, call_id: str) -> None:
        with self._lock:
            reservation = self.call_reservations.get(call_id)
            if reservation is None:
                raise AgentContextError("Unknown tool reservation")
            if reservation["outcome"] != "pending":
                return
            reservation["outcome"] = "unknown"
            reservation["handler_attempts"] = None
            self.completed_call_ids.add(call_id)
            self._dispatchable_call_ids.discard(call_id)
            self._claimed_call_ids.discard(call_id)
            self.revision += 1

    def begin_model_request(self) -> int:
        with self._lock:
            self.model_requests += 1
            self.revision += 1
            return self.model_requests

    def begin_attempt(
        self,
        *,
        call_id: str,
        tool_name: str,
        max_calls_for_tool: int | None,
        category: str = "external",
        budget_group: str | None = None,
        arguments: Mapping[str, Any] | None = None,
        tool_version: str | None = None,
    ) -> None:
        call_id = self._validate_call_id(call_id)
        group = self._group(category, budget_group)
        argument_hash = self._argument_hash(arguments)
        with self._lock:
            reservation = self.call_reservations.get(call_id)
            if reservation is not None:
                if (reservation["tool_name"] != tool_name or reservation["budget_group"] != group
                        or reservation["tool_version"] is not None and reservation["tool_version"] != tool_version
                        or reservation["argument_hash"] is not None and reservation["argument_hash"] != argument_hash):
                    raise AgentContextError("Tool call does not match its saved reservation")
                if reservation["outcome"] != "pending" or call_id in self._claimed_call_ids:
                    raise DuplicateToolCallError()
                if call_id not in self._dispatchable_call_ids:
                    raise AgentContextError("Unresolved restored tool call requires outcome-unknown recovery")
                self._claimed_call_ids.add(call_id)
                self.revision += 1
                return
            self._charge_call(call_id, tool_name, max_calls_for_tool, group)

    def next_reference_id(self) -> str:
        with self._lock:
            return f"R{len(self.references) + 1}"

    def commit_result(
        self,
        reference: Evidence | None,
        *,
        observation_size: int,
        sensitive_values: Sequence[str] = (),
    ) -> None:
        if observation_size < 0:
            raise ValueError("observation_size cannot be negative")
        with self._lock:
            self.revision += 1
            expected = f"R{len(self.references) + 1}"
            if reference is not None and reference.reference_id != expected:
                raise ValueError("reference sequence is inconsistent")
            if reference is not None and len(self.references) >= self.max_references:
                self.stop_reason = "reference_limit"
                self.limit_message = f"证据来源数量已达到上限（{self.max_references}）。"
                raise AgentLimitReachedError(self.limit_message)
            if self.max_observation_bytes is not None and self.observation_bytes + observation_size > self.max_observation_bytes:
                self.stop_reason = "observation_limit"
                self.limit_message = f"本次工具结果超过剩余观察额度（已用 {self.observation_bytes}/{self.max_observation_bytes} 字节）。"
                raise AgentLimitReachedError(self.limit_message)
            normalized_sensitive: list[str] = []
            for value in sensitive_values:
                if not isinstance(value, str) or not 3 <= len(value) <= 2048:
                    raise ValueError("sensitive observation values are invalid")
                normalized_sensitive.append(value)
            if reference is not None:
                self.references.append(reference)
            self.observation_bytes += observation_size
            self.sensitive_values.update(normalized_sensitive)

    def references_snapshot(self) -> tuple[Evidence, ...]:
        with self._lock:
            return tuple(self.references)

    def sensitive_values_snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self.sensitive_values))

    def snapshot(self) -> dict[str, Any]:
        """Private checkpoint data; callers must never project this to events."""
        with self._lock:
            # This is a projection of the attempt owners, not a driver request
            # counter. A lost tool result makes the aggregate unknown as well.
            handler_attempts = {**self._ordinary_handler_attempts, **{
                key: value["handler_attempts"] for key, value in self.call_reservations.items()}}
            known_attempts = sum(value for value in handler_attempts.values() if value is not None)
            return {"revision": self.revision, "tool_call_count": self.tool_call_count,
                    "limits": {name: getattr(self, name) for name in _LIMIT_FIELDS},
                    "logical_tool_calls": self.tool_call_count, "model_requests": self.model_requests,
                    "retry_slots_reserved": self.retry_reserved_count,
                    "call_reservations": {key: dict(value) for key, value in self.call_reservations.items()},
                    "tool_execution_attempts_by_call": handler_attempts,
                    "tool_execution_attempts": None if any(value is None for value in handler_attempts.values()) else known_attempts,
                    "known_tool_execution_attempts": known_attempts,
                    "observation_bytes": self.observation_bytes, "references": list(self.references),
                    "calls_by_tool": dict(self.calls_by_tool), "calls_by_category": dict(self.calls_by_category),
                    "used_call_ids": sorted(self.used_call_ids), "completed_call_ids": sorted(self.completed_call_ids),
                    "sensitive_values": sorted(self.sensitive_values),
                    "revoked_tool_names": sorted(self.revoked_tool_names),
                    "stop_reason": self.stop_reason, "limit_message": self.limit_message}

    def public_budget(self) -> dict[str, Any]:
        """Safe live allowances, independent of model-call and token counters."""
        with self._lock:
            return {"toolCalls": self.tool_call_count, "maxToolCalls": self.max_tool_calls,
                    "observationBytes": self.observation_bytes, "maxObservationBytes": self.max_observation_bytes,
                    "externalToolCalls": self.calls_by_category.get("external", 0),
                    "maxExternalToolCalls": self.max_external_tool_calls,
                    "memoryToolCalls": self.calls_by_category.get("memory", 0),
                    "maxMemoryToolCalls": self.max_memory_tool_calls,
                    "stopReason": self.stop_reason, "limitMessage": self.limit_message,
                    "toolLimits": {name: {"used": self.calls_by_tool.get(name, 0), "limit": limit,
                        "remaining": max(0, limit - self.calls_by_tool.get(name, 0))}
                        for name, limit in self.tool_limits.items()}}

    def restore(self, value: Mapping[str, Any]) -> None:
        """Restore accounting while retaining the stricter limits and revocations."""
        if not value:
            return
        with self._lock:
            saved_limits = value.get("limits", {})
            saved_revocations = value.get("revoked_tool_names", ())
            if not isinstance(saved_limits, Mapping) or not isinstance(saved_revocations, (list, tuple, set)):
                raise AgentContextError("Saved tool budget restrictions are invalid")
            if any(not isinstance(name, str) or not name.strip() for name in saved_revocations):
                raise AgentContextError("Saved tool revocations are invalid")
            saved_revocations = set(saved_revocations)
            restrictions_changed = not saved_revocations.issubset(self.revoked_tool_names)
            self.revoked_tool_names.update(saved_revocations)
            retained_stricter_restrictions = bool(self.revoked_tool_names - saved_revocations)
            for name in _LIMIT_FIELDS:
                if name not in saved_limits:
                    continue
                saved_limit = saved_limits[name]
                if saved_limit is None and name == "max_observation_bytes":
                    retained_stricter_restrictions |= getattr(self, name) is not None
                    continue
                if (type(saved_limit) is not int or saved_limit < 0
                        or name in {"max_tool_calls", "max_observation_bytes", "max_references"} and saved_limit == 0
                        or name == "max_extra_retries" and saved_limit > 2):
                    raise AgentContextError("Saved tool budget limit is invalid")
                current_limit = getattr(self, name)
                if current_limit is None:
                    setattr(self, name, saved_limit)
                    restrictions_changed = True
                    continue
                restrictions_changed = restrictions_changed or saved_limit < current_limit
                retained_stricter_restrictions = retained_stricter_restrictions or current_limit < saved_limit
                setattr(self, name, min(current_limit, saved_limit))
            if value.get("revision", 0) < self.revision:
                if restrictions_changed:
                    self.revision += 1
                return
            self.revision = int(value["revision"]) + int(retained_stricter_restrictions)
            self.tool_call_count = int(value["tool_call_count"])
            self.model_requests = int(value.get("model_requests", 0))
            self.retry_reserved_count = int(value.get("retry_slots_reserved", 0))
            self.call_reservations = {key: dict(item) for key, item in value.get("call_reservations", {}).items()}
            for key, reservation in self.call_reservations.items():
                reservation.setdefault("retry_protocol", True)
                if not reservation["retry_protocol"] and reservation["outcome"] == "pending":
                    self._dispatchable_call_ids.add(key)
            self._ordinary_handler_attempts = {key: int(count) for key, count in value.get("tool_execution_attempts_by_call", {}).items()
                                              if key not in self.call_reservations and count is not None}
            self.observation_bytes = int(value["observation_bytes"])
            self.references = list(value.get("references", []))
            self.calls_by_tool = dict(value.get("calls_by_tool", {}))
            self.calls_by_category = dict(value.get("calls_by_category", {}))
            self.used_call_ids = set(value.get("used_call_ids", []))
            self.completed_call_ids = set(value.get("completed_call_ids", []))
            self.sensitive_values = set(value.get("sensitive_values", []))
            self.stop_reason = value.get("stop_reason")
            self.limit_message = value.get("limit_message")
            if self.stop_reason == "observation_limit" and self.max_observation_bytes is None:
                # An explicit new configuration may remove this ceiling. It
                # never removes the consumed bytes or any tool-call limit.
                self.stop_reason = self.limit_message = None

    def complete_attempt(self, call_id: str) -> None:
        with self._lock:
            self.completed_call_ids.add(call_id)
            reservation = self.call_reservations.get(call_id)
            if reservation is not None and reservation["outcome"] == "pending":
                reservation["outcome"] = "completed"
            self._dispatchable_call_ids.discard(call_id)
            self._claimed_call_ids.discard(call_id)
            self.revision += 1
