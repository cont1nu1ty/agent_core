"""Read only dependency metadata from official committed Run checkpoints."""
from collections.abc import Mapping
from agent_core.runtime.streaming import sync_facade
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph


class ContextDependencyState(TypedDict, total=False):
    state_version: str
    run_identity: dict[str, str | None]
    run_exposed_memory_sources: list[str]
    run_memory_exposure_complete: bool
    context_summary: dict[str, Any]
    context_history_dependencies: list[str]
    context_has_legacy_history: bool


async def aread_context_dependencies(checkpointer: Any, run_id: str, *, identity: Mapping[str, Any]) -> tuple[str, ...] | None:
    """Materialize through get_state; never run tools, read files or write state.

    Historical checkpoints predating exposure tracking have no dependencies to
    return. The host still owns eligibility of their canonical conversation.
    """
    from agent_core.planning.plan_execute.contracts import PLAN_STATE_VERSION

    builder = StateGraph(ContextDependencyState)
    builder.add_edge(START, END)
    graph = builder.compile(checkpointer=checkpointer)
    saved = await graph.aget_state({"configurable": {"thread_id": run_id}})
    if not saved.values:
        return None
    actual = saved.values.get("run_identity", {})
    if saved.values.get("state_version") not in {"1", "2", "plan-execute-1", "plan-execute-2", PLAN_STATE_VERSION} or any(
            actual.get(key) != value for key, value in identity.items()):
        raise PermissionError("context dependency checkpoint does not match its authorized source")
    summary = saved.values.get("context_summary") or {}
    if ("run_exposed_memory_sources" not in saved.values
            or saved.values.get("run_memory_exposure_complete") is False
            or saved.values.get("context_has_legacy_history")
            or (summary.get("text") and summary.get("has_legacy_dependencies", True))):
        return None
    # Older source Runs form an acyclic history dependency chain. In-Run
    # message sources cannot refer back to this checkpoint without a cycle.
    inherited = [*saved.values.get("context_history_dependencies", ()), *summary.get("source_ids", ())]
    return tuple(dict.fromkeys([*map(str, saved.values["run_exposed_memory_sources"]),
        *(str(source) for source in inherited if not str(source).startswith(f"{run_id}:"))]))


read_context_dependencies = sync_facade(aread_context_dependencies)
