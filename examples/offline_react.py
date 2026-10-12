"""A real ReAct graph with a deterministic model and one ordinary tool.

Run after installing the package: python examples/offline_react.py
No model service, database, credentials or test helper imports are required.
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timedelta, timezone
import json

from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from pydantic import BaseModel

from agent_core import (
    AgentConfig, AgentService, CancellationToken, RunContext, RunRequest,
    ToolExecutor, ToolRegistry, ToolSpec, WorkflowDefinition, WorkflowExecution,
    WorkflowRegistry,
)
from agent_core.context.builder import build_messages
from agent_core.memory.stores.snapshot import SnapshotHistory
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.runtime.budget import ExecutionBudget
from agent_core.tools.langchain import build_langchain_tools


class AddInput(BaseModel):
    left: int
    right: int


@dataclass(frozen=True)
class DemoToolRuntime:
    cancellation: CancellationToken
    execution: ExecutionBudget
    deadline: datetime | None = None

    def clock(self):
        return datetime.now(timezone.utc)

    def raise_if_cancelled(self):
        self.cancellation.raise_if_cancelled()

    def with_timeout(self, seconds):
        deadline = self.clock() + timedelta(seconds=seconds)
        return replace(self, deadline=min(deadline, self.deadline) if self.deadline else deadline)


class DeterministicDemoModel:
    """Fake responses exercise real routing; this is not an LLM evaluation."""

    def bind_tools(self, tools, **kwargs):
        return self

    def stream(self, messages, **kwargs):
        result = next((message.content for message in reversed(messages)
                       if isinstance(message, ToolMessage) and message.name == "add_numbers"), None)
        if result is None:
            yield AIMessage(content="", tool_calls=[{
                "name": "add_numbers", "args": {"left": 2, "right": 3},
                "id": "demo-add-1", "type": "tool_call",
            }])
        else:
            yield AIMessageChunk(content=f"2 + 3 = {result}")

    def close(self):
        pass


def serialize(value):
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"Unsupported public demo value: {type(value).__name__}")


async def main():
    runtime = DemoToolRuntime(CancellationToken(), ExecutionBudget(max_tool_calls=2))
    executor = ToolExecutor(registry=ToolRegistry([
        ToolSpec(name="add_numbers", description="Add two integers", input_schema=AddInput,
                 handler=lambda arguments, _runtime: str(arguments.left + arguments.right)),
    ]))
    policy = ReActPolicy()

    def factory(request, context, resources):
        tools = build_langchain_tools(executor.registry.definitions, executor=executor)
        graph = compile_react_graph(resources.model, tools,
            stream_chat_message=resources.stream_chat_message, policy=policy)
        return WorkflowExecution(graph=graph, initial_state={
            "messages": build_messages(request.input, SnapshotHistory()),
            "references": [], "planning_rounds": 0, "tool_call_count": 0,
            "observation_bytes": 0, "used_tool_call_ids": [],
            "stop_reason": None, "final_answer": "",
        }, context=ReActContext(request=request, cancellation=context.cancellation,
            settings=resources.settings, async_stream_runner=resources.async_stream_runner,
            clock=context.clock, tool_runtime=runtime, policy=policy))

    service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("offline_add", factory)]),
                           model_factory=DeterministicDemoModel, settings=AgentConfig(max_tool_calls=2))
    request = RunRequest("offline-demo-run", "offline_add", "Add 2 and 3 using the registered tool.")
    completed = None
    print("Offline demo: deterministic Fake model, real ReAct graph, no external services.")
    async for event in service.astream(request, RunContext(cancellation=runtime.cancellation)):
        print(json.dumps({"sequence": event.sequence, **event.as_update()},
                         ensure_ascii=False, default=serialize))
        if event.kind == "run.completed":
            completed = event.payload["result"]
    if completed is None or completed.output != "2 + 3 = 5" or runtime.execution.tool_call_count != 1:
        raise RuntimeError("Offline demo did not complete the expected governed tool call")
    print(json.dumps({"output": completed.output, "budget": runtime.execution.public_budget()},
                     ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
