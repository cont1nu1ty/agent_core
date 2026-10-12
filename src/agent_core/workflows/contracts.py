"""Factories compose executable graphs without owning a second scheduler."""
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Mapping

from agent_core.config import AgentConfig
from agent_core.contracts.context import RunContext
from agent_core.contracts.runs import RunRequest, RunResult


def default_result(state: Mapping[str, Any]) -> RunResult:
    # Graph state is private. Non-text workflows choose their public output
    # explicitly through result_factory, never by leaking all internal fields.
    return RunResult(output=state.get("final_answer"))


@dataclass(frozen=True)
class RunResources:
    model: Any
    settings: AgentConfig
    async_stream_runner: Any
    stream_chat_message: Callable[..., Any]
    checkpointer: Any = None
    store: Any = None
    astream_chat_message: Callable[..., Any] | None = None


@dataclass(frozen=True)
class PersistenceResources:
    """Borrowed official resources, scoped by the host's async context manager."""

    checkpointer: Any = None
    store: Any = None


@dataclass(frozen=True)
class WorkflowExecution:
    graph: Any
    initial_state: dict[str, Any]
    context: Any = None
    result_factory: Callable[[Mapping[str, Any]], RunResult] = default_result
    state_version: str = "1"
    prepare_resume: Callable[[Any, dict[str, Any], Any], dict[str, Any]] | None = None
    finalize_error: Callable[[Any, dict[str, Any], Any, BaseException], None] | None = None


@dataclass(frozen=True)
class WorkflowDefinition:
    name: str
    factory: Callable[[RunRequest, RunContext, RunResources], WorkflowExecution | Awaitable[WorkflowExecution]]
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.version.strip() or not callable(self.factory):
            raise ValueError("workflow requires a name, version and callable factory")
