"""Govern public community tools without replacing their implementations."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from typing import Any
from agent_core.runtime.streaming import call_async, run_sync
from agent_core.tools.contracts import AuthorizationDecision, NativeAuthorizer

from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent_core.contracts.errors import AgentContextError, AgentLimitReachedError, DuplicateToolCallError
from agent_core.contracts.events import activity_label, activity_scope


_PUBLIC_ERROR_CODES = frozenset({
    "invalid_tool_call", "invalid_tool_arguments", "unknown_tool", "tool_not_allowed",
    "tool_not_exposed", "tool_limit_reached", "tool_execution_interrupted",
    "file_not_found", "file_read_offset_out_of_range", "agent_limit_reached",
    "tool_call_limit", "per_tool_limit", "observation_limit", "reference_limit", "category_tool_limit",
    "unknown_memory_tool", "memory_scope_denied", "memory_path_denied", "memory_revision_changed",
    "memory_read_only", "memory_size_limit", "invalid_task_notes", "tool_query_failed",
})


def tool_error_code(message: ToolMessage) -> str | None:
    """Expose only a known disposition, never tool content or backend errors."""
    if message.status != "error":
        return None
    try:
        payload = json.loads(str(message.content))
    except (TypeError, ValueError):
        payload = None
    code = payload.get("errorCode") if isinstance(payload, dict) else None
    return code if isinstance(code, str) and code in _PUBLIC_ERROR_CODES else "tool_query_failed"


def _limit_receipt(error: AgentLimitReachedError, execution: Any, call_id: str, name: str) -> ToolMessage:
    return ToolMessage(content=json.dumps({"ok": False,
        "errorCode": execution.stop_reason or error.code, "message": error.public_message}, ensure_ascii=False),
        tool_call_id=call_id, name=name, status="error")


class NativeToolGovernance:
    def __init__(self, *, metadata: Mapping[str, Mapping[str, Any]] | None = None,
                 authorize: Callable[..., Any] | None = None, access: Any = None,
                 authorizer: NativeAuthorizer | None = None) -> None:
        self.metadata = dict(metadata or {})
        self.authorize = authorize
        self.access = access
        self.authorizer = authorizer

    async def aexecute(self, request: Any, handler: Callable[..., Any]) -> Command:
        context = request.runtime.context
        context.cancellation.raise_if_cancelled()
        execution = context.tool_runtime.execution
        tool = request.tool
        call_id = request.tool_call["id"]
        name = request.tool_call["name"]
        identity = {"callId": call_id, "toolName": name,
                    "activityScopeId": activity_scope(getattr(request.runtime, "state", None)),
                    "activityLabel": activity_label(getattr(request.runtime, "state", None))}
        governed = bool(tool is not None and (tool.extras or {}).get("agent_core_governed"))
        metadata = self.metadata.get(name, {})
        rejection = await self.access.acheck_call(request) if self.access is not None else None
        if rejection is not None:
            code, message = rejection
            definition = self.access.retry_definition(name)
            try:
                execution.begin_attempt(call_id=call_id, tool_name=name,
                    max_calls_for_tool=definition.max_calls_per_run if definition else None,
                    budget_group=definition.budget_group if definition else metadata.get("budget_group", "external"),
                    arguments=request.tool_call.get("args", {}),
                    tool_version=definition.version if definition else metadata.get("version"))
            except (AgentLimitReachedError, DuplicateToolCallError):
                pass
            execution.complete_attempt(call_id)
            public_name = name if code != "tool_not_exposed" else "unavailable_tool"
            request.runtime.stream_writer({"kind": "tool.completed", **identity,
                "toolName": public_name, "status": "failed", "summary": message, "errorCode": code})
            return Command(update={"messages": [ToolMessage(
                json.dumps({"ok": False, "errorCode": code, "message": message}, ensure_ascii=False),
                tool_call_id=call_id, name=public_name, status="error")], "execution_ledger": execution.snapshot(),
                **self.access.checkpoint_update()})
        if not governed:
            request.runtime.stream_writer({"kind": "tool.started", **identity})
            try:
                execution.begin_attempt(call_id=call_id, tool_name=name, max_calls_for_tool=None,
                                        budget_group=metadata.get("budget_group", "external"),
                                        arguments=request.tool_call.get("args", {}), tool_version=metadata.get("version"))
                trusted_search = self.access is not None and tool is self.access.search_tool
                decision = None
                if not trusted_search:
                    arguments = request.tool_call.get("args", {})
                    if self.authorize is not None and metadata:
                        decision = await call_async(self.authorize, name, arguments)
                    if (decision is None or decision.allowed) and self.authorizer is not None:
                        decision = await call_async(getattr(self.authorizer, "authorize", self.authorizer), tool_name=name,
                            arguments=arguments, runtime=request.runtime, metadata=metadata)
                    if decision is None:
                        decision = AuthorizationDecision(False, "tool_not_allowed", "Native tool requires explicit authorization")
                if decision is not None and not decision.allowed:
                    result = ToolMessage(content=json.dumps({"ok": False, "errorCode": decision.code,
                        "message": decision.message}, ensure_ascii=False), tool_call_id=call_id, name=name, status="error")
                else:
                    execution.record_handler_attempt(call_id)
                    result = await call_async(handler, request)
            except AgentLimitReachedError as error:
                result = _limit_receipt(error, execution, call_id, name)
        else:
            result = await call_async(handler, request)
        context.cancellation.raise_if_cancelled()
        if isinstance(result, Command):
            if result.goto or result.graph or result.resume is not None:
                raise AgentContextError("Tool Command may not change graph routing or resume state")
            if not isinstance(result.update, Mapping):
                raise AgentContextError("Tool Command requires a declared mapping update")
            update = dict(result.update)
            allowed = {"messages", *metadata.get("allowed_state_keys", ())}
            allowed.difference_update({"tool_discovery", "tool_visibility", "execution_ledger",
                                       "run_identity", "state_version", "stop_reason", "planning_rounds",
                                       "tool_call_count", "used_tool_call_ids", "final_answer", "references",
                                       "observation_bytes", "context_summary", "summary_calls",
                                       "plan_state", "model_ledger", "input_action", "route", "tool_modes"})
            if self.access is not None and tool is self.access.search_tool:
                allowed.add("tool_discovery")
            if governed:
                allowed.update({"references", "observation_bytes"})
                if self.authorize is not None:
                    allowed.add("files")  # Official FilesystemMiddleware result offload.
            if set(update) - allowed:
                raise AgentContextError("Tool Command attempted to change protected Run state")
        elif isinstance(result, ToolMessage):
            update = {"messages": [result]}
        else:
            raise AgentContextError("Tool returned neither a ToolMessage nor a legal Command")
        messages = update.get("messages", [])
        if not isinstance(messages, list) or any(not isinstance(msg, ToolMessage) or msg.tool_call_id != call_id for msg in messages):
            raise AgentContextError("Tool may only answer its own tool-call identity")
        if not governed:
            size = sum(len(str(msg.content).encode("utf-8")) for msg in messages)
            try:
                execution.commit_result(None, observation_size=size)
            except AgentLimitReachedError as error:
                update["messages"] = [_limit_receipt(error, execution, call_id, name)]
                messages = update["messages"]
                size = 0
            update["observation_bytes"] = size
            failed = any(msg.status == "error" for msg in messages)
            request.runtime.stream_writer({"kind": "tool.completed", **identity,
                                           "status": "failed" if failed else "succeeded",
                                           **({"errorCode": next(tool_error_code(msg) for msg in messages if msg.status == "error")}
                                              if failed else {})})
        execution.complete_attempt(call_id)
        update["execution_ledger"] = execution.snapshot()
        if self.access is not None:
            update.update(self.access.checkpoint_update())
        return Command(update=update)

    def execute(self, request, handler):
        return run_sync(self.aexecute(request, handler))
