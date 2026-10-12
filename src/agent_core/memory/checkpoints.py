"""Read task memory with the community graph's channel materialization."""

from __future__ import annotations

from collections.abc import Mapping
from agent_core.runtime.streaming import sync_facade
from typing import Any

from deepagents.middleware.filesystem import FilesystemState
from langchain.agents.middleware.todo import PlanningState
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from agent_core.memory.components import TaskMemoryState


class TaskCheckpointState(FilesystemState, PlanningState, TaskMemoryState):
    """The same official channels used by the executing memory components."""

    plan_state: dict[str, Any]
    references: list[Any]


async def aread_task_checkpoint(checkpointer: BaseCheckpointSaver,
                         pointer: Mapping[str, Any]) -> dict[str, Any]:
    """Materialize a root checkpoint without constructing or running an agent.

    The host must authorize the pointer and validate its application version.
    An exact checkpoint ID selects committed state at that point. A thread-only
    pointer reads the latest available state for a live projection. This reader
    never resumes unfinished tools or rewrites a checkpoint.
    """
    thread_id = pointer.get("thread_id")
    checkpoint_id = pointer.get("checkpoint_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("task checkpoint requires a thread_id")
    if checkpoint_id is not None and (not isinstance(checkpoint_id, str) or not checkpoint_id):
        raise ValueError("task checkpoint_id must be a nonempty string")
    if pointer.get("checkpoint_ns", ""):
        raise ValueError("task memory requires a root checkpoint")
    configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id

    # FilesystemState contains the official DeltaChannel. Raw saver values omit
    # these files; get_state delegates their reconstruction to LangGraph.
    builder = StateGraph(TaskCheckpointState)
    builder.add_edge(START, END)
    graph = builder.compile(checkpointer=checkpointer)
    saved = await graph.aget_state({"configurable": configurable})
    if saved.created_at is None and checkpoint_id is not None:
        raise ValueError("task checkpoint is unavailable")
    return {key: saved.values[key] for key in ("files", "todos", "task_source_ids", "plan_state", "references")
            if key in saved.values}


read_task_checkpoint = sync_facade(aread_task_checkpoint)
