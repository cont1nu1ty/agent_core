"""Expose governed tools to LangChain without leaking injected runtime inputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Annotated, Any

from langchain.tools import ToolRuntime as LangChainToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from langgraph.types import Command
from pydantic import BaseModel, ConfigDict, PlainSerializer

from agent_core.tools.contracts import ToolResult, ToolRuntime, ToolSpec
from agent_core.tools.executor import ToolExecutor
from agent_core.runtime.streaming import run_sync
from agent_core.contracts.events import activity_label, activity_scope


class ToolEventPolicy:
    """Public event projection; applications may supply their own wording/schema."""

    def failure_summary(self) -> str:
        return "Tool execution failed"

    def validate_result(self, result: ToolResult) -> None:
        pass

    def success_summary(self, result: ToolResult) -> str:
        return "Tool execution completed"

    def reference_payload(self, reference: Any) -> dict[str, Any]:
        if isinstance(reference, BaseModel):
            return reference.model_dump(by_alias=True, mode="json")
        if is_dataclass(reference):
            return asdict(reference)
        return {"referenceId": reference.reference_id}


_InjectedRuntime = Annotated[Any, InjectedToolArg, PlainSerializer(lambda _value: "<injected>", return_type=str)]


class _ExecutorRuntimeInput(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)
    runtime: _InjectedRuntime


class _GovernedStructuredTool(StructuredTool):
    @property
    def tool_call_schema(self):
        schema = (self.extras or {}).get("advertised_schema")
        return super().tool_call_schema if schema is None else schema


def _trusted_agent_runtime(runtime: LangChainToolRuntime) -> ToolRuntime:
    candidate = getattr(runtime.context, "tool_runtime", None)
    if candidate is None:
        raise RuntimeError("tool execution requires server-injected context.tool_runtime")
    return candidate


def build_langchain_tool(definition: ToolSpec, *, executor: ToolExecutor, event_policy: ToolEventPolicy | None = None) -> BaseTool:
    policy = event_policy or ToolEventPolicy()

    async def ainvoke(runtime: LangChainToolRuntime, **arguments: Any) -> Command:
        call_id = runtime.tool_call_id
        if not call_id:
            raise RuntimeError("tool call is missing its call id")
        identity = {"callId": call_id, "toolName": definition.name,
                    "activityScopeId": activity_scope(getattr(runtime, "state", None)),
                    "activityLabel": activity_label(getattr(runtime, "state", None))}
        runtime.stream_writer({"kind": "tool.started", **identity})
        try:
            result = await executor.aexecute(call_id=call_id, tool_name=definition.name, arguments=arguments, runtime=_trusted_agent_runtime(runtime))
        except Exception:
            runtime.stream_writer({"kind": "tool.completed", **identity, "status": "failed", "summary": policy.failure_summary()})
            raise
        if not result.succeeded:
            runtime.stream_writer({"kind": "tool.completed", **identity, "status": "failed", "summary": result.message})
            return Command(update={"messages": [ToolMessage(content=result.content, name=definition.name, tool_call_id=call_id, status="error")]})
        policy.validate_result(result)
        if result.reference is not None:
            runtime.stream_writer({"kind": "reference.added", "callId": call_id, "payload": policy.reference_payload(result.reference)})
        runtime.stream_writer({"kind": "tool.completed", **identity, "status": "succeeded", "summary": policy.success_summary(result)})
        update: dict[str, Any] = {
            "messages": [ToolMessage(content=result.content, name=definition.name, tool_call_id=call_id, status="success")],
            "observation_bytes": result.content_size,
        }
        if result.reference is not None:
            update["references"] = [result.reference]
        return Command(update=update)

    def invoke(runtime: LangChainToolRuntime, **arguments: Any) -> Command:
        return run_sync(ainvoke(runtime, **arguments))

    return _GovernedStructuredTool.from_function(
        func=invoke, coroutine=ainvoke, name=definition.name, description=definition.description,
        args_schema=_ExecutorRuntimeInput, infer_schema=False,
        extras={"advertised_schema": definition.input_schema, "agent_core_governed": True,
                "agent_core_max_calls": definition.max_calls_per_run,
                "agent_core_budget_group": definition.budget_group,
                "agent_core_definition": definition, "agent_core_executor": executor},
    )


def build_langchain_tools(definitions: Sequence[ToolSpec], *, executor: ToolExecutor, event_policy: ToolEventPolicy | None = None) -> tuple[BaseTool, ...]:
    return tuple(build_langchain_tool(item, executor=executor, event_policy=event_policy) for item in definitions)
