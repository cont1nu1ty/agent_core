"""Translate graph output into public events, without exposing graph state."""
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from dataclasses import replace
from typing import Any
import json
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from agent_core.runtime.streaming import call_async, iter_sync
from agent_core.contracts.context import RunContext
from agent_core.contracts.artifacts import Artifact
from agent_core.contracts.events import RunEvent, activity_label, activity_scope
from agent_core.contracts.runs import RunRequest
from agent_core.contracts.errors import AgentContextError
from agent_core.memory.notes import task_memory_snapshot
from agent_core.memory.binding import restore_memory_binding
from agent_core.tools.native import tool_error_code
from agent_core.workflows.contracts import WorkflowExecution


def _tool_messages(value: Any) -> Iterator[ToolMessage]:
    if isinstance(value, ToolMessage):
        if not value.additional_kwargs.get("agent_output_schema"):
            yield value
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _tool_messages(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _tool_messages(item)


async def astream_graph(execution: WorkflowExecution, request: RunRequest, context: RunContext,
                 *, recursion_limit: int) -> AsyncIterator[RunEvent]:
    sequence = 0
    config: dict[str, Any] = {"recursion_limit": recursion_limit, "configurable": {"thread_id": request.run_id}}
    initial_state: dict[str, Any] = {**execution.initial_state, "state_version": execution.state_version,
                                   "run_identity": {"run_id": request.run_id, "workflow_id": request.workflow_id, "binding_fingerprint": request.binding_fingerprint}}
    memory = getattr(execution.context, "memory", None)
    if memory is not None:
        if memory.scope.run_id != request.run_id:
            raise AgentContextError("Memory binding belongs to a different Run")
        initial_state["run_identity"].update({name: getattr(memory.scope, name) for name in
            ("tenant_id", "user_id", "project_id", "session_id", "task_id")})
    persisted = bool(getattr(execution.graph, "checkpointer", None))
    if request.resume and not persisted:
        raise AgentContextError("Cannot resume without the configured checkpoint saver")
    if persisted:
        saved = await execution.graph.aget_state(config)
        if request.resume:
            if not saved.values:
                raise AgentContextError("Saved Run checkpoint is missing; automatic rerun is disabled")
            if saved.values.get("state_version") != execution.state_version:
                raise AgentContextError("Saved Run checkpoint version is incompatible")
            if saved.values.get("run_identity") != initial_state["run_identity"]:
                raise AgentContextError("Saved Run checkpoint identity does not match")
            if request.checkpoint_id and saved.config["configurable"].get("checkpoint_id") != request.checkpoint_id:
                raise AgentContextError("Saved Run checkpoint changed; refusing to replay an old branch")
            if execution.prepare_resume is not None:
                config = await call_async(execution.prepare_resume, execution.graph, config, saved)
                # Fetch latest without an exact checkpoint ID so successful
                # pending writes are included by the public get_state API.
                config = {**config, "recursion_limit": recursion_limit,
                          "configurable": {"thread_id": request.run_id}}
                saved = await execution.graph.aget_state(config)
            initial_state = dict(saved.values)
            if memory is not None and not restore_memory_binding(memory, initial_state):
                raise AgentContextError("Saved Run memory release is missing; cannot restore tool scope")
            runtime_budget = getattr(getattr(execution.context, "tool_runtime", None), "execution", None)
            if runtime_budget is not None:
                runtime_budget.restore(initial_state.get("execution_ledger", {}))
        elif saved.values:
            raise AgentContextError("Run identity already has saved execution; use explicit resume")
    final_state: dict[str, Any] = initial_state
    steps: dict[str, str] = {}
    call_scopes: dict[str, str] = {}
    call_labels: dict[str, str] = {}
    completed_calls: set[str] = set()
    task_projection: dict[str, Any] | None = None
    suspended = False
    graph_input = None if request.resume else initial_state
    if request.resume_decision is not None:
        decision = request.resume_decision
        current_id = saved.config["configurable"].get("checkpoint_id")
        interrupts = {item.id for item in saved.interrupts}
        if current_id != decision.checkpoint_id or set(decision.decisions) != interrupts or not interrupts:
            raise AgentContextError("Resume decisions do not match the saved pause and checkpoint")
        graph_input = Command(resume=decision.decisions)
    elif request.resume and saved.interrupts:
        raise AgentContextError("Paused Run requires decisions for its saved interrupts")
    if request.resume:
        context.cancellation.raise_if_cancelled()
        sequence += 1
        yield RunEvent(request.run_id, request.attempt, sequence, "run.resumed", {
            "checkpoint": {"thread_id": request.run_id,
                "checkpoint_id": saved.config["configurable"].get("checkpoint_id"),
                "state_version": execution.state_version}})
    stream = execution.graph.astream(
        graph_input, config=config,
        context=execution.context, stream_mode=["updates", "custom", "values"], version="v2",
        # A subsequent node must not start before its predecessor checkpoint
        # commits. Hosts still project only committed checkpoint pointers.
        **({"durability": "sync"} if persisted else {}),
    )
    stream_error: BaseException | None = None
    try:
        async for part in stream:
            context.cancellation.raise_if_cancelled()
            kind, data = part.get("type"), part.get("data")
            if kind == "values":
                if isinstance(data, Mapping):
                    final_state = dict(data)
                    if "files" in data or "todos" in data or "plan_state" in data:
                        projection = task_memory_snapshot(data)
                        if projection != task_projection:
                            task_projection = projection
                            sequence += 1
                            yield RunEvent(request.run_id, request.attempt, sequence, "task.memory", projection)
                continue
            events = []
            if kind == "custom" and isinstance(data, Mapping):
                events.append((str(data.get("kind") or "workflow.event"),
                               {key: value for key, value in data.items() if key != "kind"}))
            elif kind == "updates":
                if isinstance(data, Mapping) and "__interrupt__" in data:
                    suspended = True
                # ToolNode validation failures do not pass through the executor.
                for message in _tool_messages(data):
                    if message.tool_call_id and message.tool_call_id not in completed_calls:
                        failed = message.status == "error" or '"ok":false' in str(message.content).replace(" ", "").lower()
                        events.append(("tool.observed", {"callId": message.tool_call_id,
                            "toolName": message.name, "status": "failed" if failed else "succeeded",
                            "activityScopeId": call_scopes.get(message.tool_call_id, activity_scope(final_state)),
                            "activityLabel": call_labels.get(message.tool_call_id, activity_label(final_state)),
                            **({"errorCode": tool_error_code(message) or "tool_query_failed"} if failed else {})}))
            for event_kind, payload in events:
                nested = payload.get("payload", {})
                call_id = str(payload["callId"]) if payload.get("callId") else None
                step_id = payload.get("stepId")
                scope_id = payload.get("activityScopeId")
                label = payload.get("activityLabel")
                if isinstance(nested, Mapping):
                    step_id = step_id or nested.get("stepId")
                    scope_id = scope_id or nested.get("activityScopeId")
                    label = label or nested.get("activityLabel")
                step_id = str(step_id) if step_id else steps.get(call_id)
                scope_id = str(scope_id) if scope_id else call_scopes.get(call_id, activity_scope(final_state))
                label = str(label) if label else call_labels.get(call_id, activity_label(final_state))
                if call_id and step_id:
                    steps[call_id] = step_id
                if call_id:
                    call_scopes[call_id] = scope_id
                    call_labels[call_id] = label
                if call_id and event_kind in {"tool.completed", "tool.observed"}:
                    completed_calls.add(call_id)
                sequence += 1
                yield RunEvent(request.run_id, request.attempt, sequence, event_kind,
                               payload, step_id=step_id, call_id=call_id, activity_scope_id=scope_id,
                               activity_label=label)
        context.cancellation.raise_if_cancelled()
        checkpoint: dict[str, Any] | None = None
        if persisted:
            saved = await execution.graph.aget_state(config)
            final_state = dict(saved.values)
            checkpoint = {"thread_id": request.run_id,
                          "checkpoint_id": saved.config["configurable"].get("checkpoint_id"),
                          "state_version": execution.state_version}
            suspended = suspended or bool(saved.interrupts)
        sequence += 1
        if suspended:
            yield RunEvent(request.run_id, request.attempt, sequence, "run.suspended", {"checkpoint": checkpoint,
                "interruptions": [{"id": item.id, "value": json.loads(json.dumps(item.value, ensure_ascii=False, allow_nan=False))} for item in saved.interrupts] if persisted else []})
            return
        result = execution.result_factory(final_state)
        for index, artifact in enumerate(result.artifacts):
            if not isinstance(artifact, Artifact):
                continue  # Existing host Evidence retains its reference projection.
            yield RunEvent(request.run_id, request.attempt, sequence, "artifact.ready", {
                "index": index, "artifact": {"kind": artifact.kind,
                    "value": artifact.value, "metadata": dict(artifact.metadata)}})
            sequence += 1
        metadata = dict(result.metadata)
        if checkpoint is not None:
            metadata["checkpoint"] = checkpoint
        if "files" in final_state or "todos" in final_state or "plan_state" in final_state:
            metadata["task_memory"] = task_memory_snapshot(final_state)
        metadata["stop_reason"] = final_state.get("stop_reason")
        yield RunEvent(request.run_id, request.attempt, sequence, "run.completed",
                       {"result": replace(result, metadata=metadata)})
    except BaseException as error:
        stream_error = error
        raise
    finally:
        close = getattr(stream, "aclose", None)
        if callable(close):
            try:
                await close()
            except Exception:
                # Cleanup must preserve the run's result or original failure.
                pass
        if stream_error is not None and persisted and execution.finalize_error is not None:
            # Close first: the workflow may persist shared meters only after
            # its active model/tool boundary has stopped changing them.
            try:
                await call_async(execution.finalize_error, execution.graph, config, execution.context, stream_error)
            except Exception as finalization_error:
                stream_error.add_note(
                    f"Workflow failure-state persistence failed: {type(finalization_error).__name__}"
                )


def stream_graph(execution, request, context, *, recursion_limit):
    """Compatibility facade; execution semantics live in astream_graph."""
    yield from iter_sync(astream_graph(execution, request, context, recursion_limit=recursion_limit))
