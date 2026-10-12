"""Root channels use upstream reducers for messages, files and task memory."""

from typing import Annotated, Any

from deepagents.middleware.filesystem import FilesystemState
from deepagents.middleware.memory import MemoryState
from deepagents.middleware.skills import SkillsState
from langchain.agents.middleware.todo import PlanningState

from agent_core.memory.components import TaskMemoryState
from agent_core.planning.react.state import ReActState, merge_execution_ledger


class PlanExecuteState(ReActState, FilesystemState, PlanningState, TaskMemoryState, MemoryState, SkillsState):
    plan_state: dict[str, Any]
    input_action: dict[str, Any]
    tool_modes: dict[str, bool]
    plan_stage: str
    model_ledger: Annotated[dict[str, Any], merge_execution_ledger]
    artifact_kind: str
    stage_output: dict[str, Any]
    stage_failure: str | None
    route: str
    plan_rubric: dict[str, Any]
    execution_answer: str


__all__ = ["PlanExecuteState"]
