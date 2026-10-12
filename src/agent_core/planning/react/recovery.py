"""Public graph recovery adaptation for isolated tool Send state snapshots."""

from collections.abc import Mapping
import json
from typing import Any
from agent_core.runtime.streaming import run_sync

from langchain_core.messages import ToolMessage

from agent_core.runtime.budget import ExecutionBudget


async def aprepare_react_resume(graph: Any, config: dict[str, Any], snapshot: Any) -> dict[str, Any]:
    """Commit successful sibling writes before redispatching unfinished tools.

    LangGraph preserves successful Send results, but an unfinished sibling's
    StateBackend reader still sees the old superstep. Public update_state creates
    a checkpoint from the merged snapshot. The owned after_model node's standard
    pending-call routing excludes calls that already have a ToolMessage.

    Additive references/observation channels are deliberately not copied: the
    official update_state already applies successful pending writes once.
    """
    tool_tasks = [task for task in snapshot.tasks if task.name == "tools"]
    if any(task.interrupts for task in tool_tasks):
        return config  # Preserve official interrupt IDs for a future host resume flow.
    partial = any(task.result is not None for task in tool_tasks) and any(task.result is None for task in tool_tasks)
    if "tools" not in snapshot.next:
        return config
    node = "ReActMiddleware.after_model"
    if node not in graph.get_graph().nodes:
        raise ValueError("The registered ReAct recovery node is unavailable")
    values = snapshot.values
    if not isinstance(values, Mapping):
        raise ValueError("ReAct recovery requires a mapping state")
    ledger = values.get("execution_ledger", {})
    # This adapter materializes saved accounting; current host restrictions are
    # applied by the resumed runtime, not the default values of this helper.
    saved_limits = ledger.get("limits", {})
    budget = ExecutionBudget(**{name: saved_limits[name] for name in (
        "max_tool_calls", "max_external_tool_calls", "max_memory_tool_calls", "max_discovery_tool_calls",
        "max_retry_attempts", "max_extra_retries", "max_observation_bytes", "max_references") if name in saved_limits})
    budget.restore(ledger)
    answered = {message.tool_call_id for message in values.get("messages", ()) if isinstance(message, ToolMessage)}
    unsettled = [call_id for call_id in budget.pending_call_ids() if call_id not in answered]
    if not partial and not unsettled:
        return config
    keys = ("files", "messages", "execution_ledger", "todos", "task_source_ids", "tool_discovery", "tool_visibility")
    updates = {key: values[key] for key in keys if key in values}
    if unsettled:
        messages = list(values.get("messages", ()))
        for call_id in unsettled:
            reservation = budget.call_reservations[call_id]
            budget.mark_outcome_unknown(call_id)
            messages.append(ToolMessage(
                json.dumps({"ok": False, "errorCode": "tool_execution_interrupted", "retryable": False,
                    "message": "工具执行中断，结果尚未确认；原调用不会重放。需要时可发起新的只读查询并使用剩余额度。"},
                    ensure_ascii=False), tool_call_id=call_id, name=reservation["tool_name"],
                id=f"interrupted-{call_id}", status="error"))
        updates.update(messages=messages, execution_ledger=budget.snapshot())
    return await graph.aupdate_state(config, updates, as_node=node)


def prepare_react_resume(graph, config, snapshot):
    return run_sync(aprepare_react_resume(graph, config, snapshot))

prepare_react_resume.__async__ = aprepare_react_resume
