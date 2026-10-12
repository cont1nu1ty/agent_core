"""Validate and execute one registered tool with application-supplied policy."""

from __future__ import annotations

import asyncio
from agent_core.runtime.streaming import call_port, call_async, run_sync
import inspect
import json
import math
import re
from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from pydantic import ValidationError

from agent_core.contracts.errors import AgentCancelled, AgentLimitReachedError, DuplicateToolCallError, ToolNotRegisteredError, ToolQueryTimeoutError
from agent_core.tools.contracts import ToolAuthorizer, ToolResult, ToolRuntime
from agent_core.tools.policy import AllowRegisteredTools
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.retry import RetryController


def _validation_details(error: ValidationError, schema: type[Any]) -> dict[str, Any]:
    """Use public schema facts and declared corrections, never exception text."""
    from agent_core.tools.validation import STANDARD_VALIDATION_CODES, ToolValidationHint

    document = schema.model_json_schema()
    declared_hints = getattr(schema, "tool_validation_hints", {})

    def resolve(value: dict[str, Any]) -> dict[str, Any]:
        reference = value.get("$ref", "")
        return document.get("$defs", {}).get(reference.removeprefix("#/$defs/"), value) if reference.startswith("#/$defs/") else value

    def safe_constraints(value: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        kind = value.get("type")
        if isinstance(kind, str) and kind in {"string", "number", "integer", "boolean", "array", "object", "null"}:
            result["type"] = kind
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength", "minItems", "maxItems"):
            number = value.get(key)
            if type(number) in {int, float} and math.isfinite(number):
                result[key] = number
        choices = value.get("enum")
        if isinstance(choices, list) and len(choices) <= 12 and all(
            item is None or isinstance(item, bool) or isinstance(item, str) and len(item) <= 100
            or type(item) in {int, float} and math.isfinite(item) for item in choices
        ):
            result["enum"] = list(choices)
        return result

    errors = []
    for item in error.errors(include_input=False, include_context=False, include_url=False)[:8]:
        value, parts = document, []
        for part in item.get("loc", ())[:8]:
            value = resolve(value)
            properties = value.get("properties", {})
            if isinstance(part, str) and part in properties:
                parts.append(part)
                value = properties[part]
            elif isinstance(part, int) and isinstance(value.get("items"), dict):
                parts.append(str(part))
                value = value["items"]
            else:
                parts.append("<field>")
                value = {}
        path = ".".join(parts) or "<root>"
        code = str(item.get("type", "invalid"))
        hint = declared_hints.get(code) if isinstance(declared_hints, Mapping) else None
        trusted = (code not in STANDARD_VALIDATION_CODES and re.fullmatch(r"[a-z0-9_]{1,64}", code)
                   and isinstance(hint, ToolValidationHint)
                   and all(name in document.get("properties", {}) for name in hint.fields))
        detail = {"field": path[:160], "code": code if code in STANDARD_VALIDATION_CODES or trusted else "invalid"}
        if trusted:
            detail.update(field=hint.fields[0], fields=list(hint.fields), hint=hint.hint)
            value = document["properties"][hint.fields[0]]
        constraints = safe_constraints(resolve(value))
        if constraints:
            detail["constraints"] = constraints
        errors.append(detail)
    return {"errors": errors}


class ToolExecutionPolicy:
    def call_scope(self, runtime: ToolRuntime, timeout_seconds: float) -> AbstractContextManager[ToolRuntime]:
        return nullcontext(runtime.with_timeout(timeout_seconds))

    def is_timeout(self, error: Exception) -> bool:
        return isinstance(error, TimeoutError)

    def failure(self, *, tool: str, code: str, message: str, version: str | None = None, retryable: bool = False,
                details: Mapping[str, Any] | None = None) -> ToolResult:
        data = {"ok": False, "tool": tool, "toolVersion": version, "errorCode": code, "message": message, "retryable": retryable}
        if details is not None:
            data["details"] = dict(details)
        return ToolResult(content=json.dumps(data, ensure_ascii=False, separators=(",", ":")), succeeded=False, message=message, data=data)


class ToolExecutor:
    def __init__(self, *, registry: ToolRegistry, authorizer: ToolAuthorizer | None = None, policy: ToolExecutionPolicy | None = None,
                 retry_controller: RetryController | None = None) -> None:
        self.registry = registry
        self.authorizer = authorizer or AllowRegisteredTools()
        self.policy = policy or ToolExecutionPolicy()
        self.retry_controller = retry_controller or RetryController()

    async def aexecute(self, *, call_id: str, tool_name: str, arguments: Mapping[str, Any], runtime: ToolRuntime) -> ToolResult:
        failure = self.policy.failure
        try:
            definition = self.registry.get(tool_name)
        except ToolNotRegisteredError as error:
            runtime.raise_if_cancelled()
            try:
                runtime.execution.begin_attempt(call_id=call_id, tool_name="unregistered_tool", max_calls_for_tool=None)
            except (AgentLimitReachedError, DuplicateToolCallError) as limit:
                return failure(tool="unregistered_tool", code=limit.code, message=limit.public_message)
            return failure(tool="unregistered_tool", code=error.code, message=error.public_message)

        validation_details = None
        try:
            parsed = definition.input_schema.model_validate(dict(arguments))
        except ValidationError as error:
            parsed = None
            validation_details = _validation_details(error, definition.input_schema)
        runtime.raise_if_cancelled()
        decision = await call_port(self.authorizer, "authorize", definition, runtime)
        try:
            if (runtime.execution.retries_enabled and definition.retry_policy.mode == "safe_read"
                    and call_id not in runtime.execution.used_call_ids):
                runtime.execution.reserve_call(call_id=call_id, tool_name=definition.name,
                    max_calls_for_tool=definition.max_calls_per_run, budget_group=definition.budget_group,
                    max_extra_retries=definition.retry_policy.max_extra_retries,
                    arguments=arguments, tool_version=definition.version)
            runtime.execution.begin_attempt(call_id=call_id, tool_name=definition.name,
                                            max_calls_for_tool=definition.max_calls_per_run,
                                            budget_group=definition.budget_group,
                                            arguments=arguments, tool_version=definition.version)
        except (AgentLimitReachedError, DuplicateToolCallError) as error:
            return failure(tool=definition.name, version=definition.version, code=error.code, message=error.public_message)
        if not decision.allowed:
            runtime.execution.revoke_tool(definition.name)
            return failure(tool=definition.name, version=definition.version, code=decision.code, message=decision.message)
        if parsed is None:
            return failure(tool=definition.name, version=definition.version, code="invalid_tool_arguments",
                           message="工具参数无效，请修正后重试", details=validation_details)

        timeout_seconds = definition.timeout_seconds
        if runtime.deadline is not None:
            remaining = (runtime.deadline - runtime.clock()).total_seconds()
            if remaining <= 0:
                raise ToolQueryTimeoutError("tool execution deadline exceeded")
            timeout_seconds = min(timeout_seconds, remaining)
        deadline = self.retry_controller.monotonic() + timeout_seconds
        try:
            with self.policy.call_scope(runtime, timeout_seconds) as call_runtime:
                attempt = 0
                while True:
                    self.retry_controller.remaining(call_runtime, deadline)
                    if attempt:
                        decision = await call_port(self.authorizer, "authorize", definition, call_runtime)
                        if not decision.allowed:
                            runtime.execution.revoke_tool(definition.name)
                            return failure(tool=definition.name, version=definition.version,
                                           code=decision.code, message=decision.message)
                    attempt = runtime.execution.record_handler_attempt(call_id)
                    try:
                        async with asyncio.timeout(self.retry_controller.remaining(call_runtime, deadline)):
                            raw = await call_async(definition.handler, parsed, call_runtime)
                    except (AgentCancelled, ToolQueryTimeoutError, TimeoutError):
                        raise
                    except Exception as error:
                        # Check the shared deadline before interpreting a transport
                        # timeout as a potentially recoverable stage failure.
                        self.retry_controller.remaining(call_runtime, deadline)
                        mapped = definition.recoverable_error_mapper(error) if definition.recoverable_error_mapper is not None else None
                        if mapped is None:
                            if self.policy.is_timeout(error):
                                raise ToolQueryTimeoutError("tool execution deadline exceeded") from error
                            raise
                        retry = (runtime.execution.retries_enabled and definition.retry_policy.mode == "safe_read"
                                 and mapped.failure_kind == "transient" and mapped.retryable
                                 and attempt <= runtime.execution.retry_allowance(call_id))
                        if retry and await self.retry_controller.await_before_retry(call_runtime, deadline=deadline,
                                retry_number=attempt, retry_after_seconds=mapped.retry_after_seconds):
                            continue
                        return failure(tool=definition.name, version=definition.version, code=mapped.code,
                                       message=mapped.message, retryable=mapped.retryable)
                    self.retry_controller.remaining(call_runtime, deadline)
                    break
        except TimeoutError as error:
            raise ToolQueryTimeoutError("tool execution deadline exceeded") from error
        if definition.result_builder is not None:
            result = await call_async(definition.result_builder, parsed, raw, call_runtime)
        elif isinstance(raw, ToolResult):
            result = raw
        else:
            result = ToolResult(content=raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, separators=(",", ":")), succeeded=True, data=raw)
        if result.succeeded:
            try:
                runtime.execution.commit_result(result.reference, observation_size=result.content_size, sensitive_values=result.sensitive_values)
            except AgentLimitReachedError as error:
                return failure(tool=definition.name, version=definition.version, code=error.code, message=error.public_message)
        return result

    def execute(self, **kwargs) -> ToolResult:
        return run_sync(self.aexecute(**kwargs))
