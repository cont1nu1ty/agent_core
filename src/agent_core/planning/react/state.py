"""Checkpointable ReAct state; product sessions are owned by the caller."""

from __future__ import annotations

import operator
from typing import Annotated
from typing import Any
from typing_extensions import NotRequired

from langchain.agents import AgentState

from agent_core.contracts.artifacts import Evidence

REACT_STATE_VERSION = "2"


def merge_execution_ledger(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Serial execution yields cumulative snapshots, including pending writes.

    Parallel graph tasks may deliver these snapshots in a different order.
    The monotonic revision identifies the most recent committed accounting.
    """
    return right if right.get("revision", 0) >= left.get("revision", 0) else left


class ReActState(AgentState):
    state_version: str
    run_identity: dict[str, str | None]
    execution_ledger: Annotated[dict[str, Any], merge_execution_ledger]
    model_ledger: Annotated[dict[str, Any], merge_execution_ledger]
    references: Annotated[list[Evidence], operator.add]
    planning_rounds: int
    structured_output_retries: int
    tool_call_count: int
    observation_bytes: Annotated[int, operator.add]
    used_tool_call_ids: list[str]
    stop_reason: str | None
    final_answer: str
    activity_scope_id: NotRequired[str]
    activity_label: NotRequired[str]
    context_summary: NotRequired[dict[str, Any]]
    summary_calls: NotRequired[int]
    context_warning: NotRequired[str]
    context_history_dependencies: NotRequired[list[str]]
    context_has_legacy_history: NotRequired[bool]
    tool_discovery: Annotated[NotRequired[dict[str, Any]], merge_execution_ledger]
    tool_visibility: NotRequired[dict[str, Any]]


__all__ = ["REACT_STATE_VERSION", "ReActState"]
