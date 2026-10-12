"""Execution limits supplied by the application, independent of environment names."""
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    max_planning_rounds: int = 6
    max_task_planning_rounds: int = 20
    max_tool_calls: int = 24
    max_external_tool_calls: int = 8
    max_memory_tool_calls: int = 16
    max_observation_bytes: int | None = 96 * 1024
    recursion_limit: int = 200
    max_answer_characters: int = 200_000
    max_input_tokens: int | None = 16_384
    output_token_reserve: int = 4_096
    model_context_window: int | None = None
    max_plan_steps: int = 8
    max_plan_replans: int = 3
    max_plan_model_calls: int | None = 64
    max_plan_rubric_iterations: int = 3
    context_summary_enabled: bool = False
    session_summary_enabled: bool = False
    context_summary_trigger_fraction: float = 0.7
    context_summary_max_output_tokens: int = 1_536
    context_summary_max_calls: int = 4
    tool_discovery_enabled: bool = False
    tool_retry_enabled: bool = False
    max_tool_extra_retries: int = 2
    max_tool_retry_slots: int = 8
    max_discovery_calls: int = 3
    max_tool_schema_tokens: int = 4096

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in {"context_summary_enabled", "session_summary_enabled", "tool_discovery_enabled", "tool_retry_enabled"}:
                if type(value) is not bool:
                    raise ValueError(f"{name} must be boolean")
                continue
            if name in {"max_tool_extra_retries", "max_tool_retry_slots"}:
                if type(value) is not int or value < 0 or (name == "max_tool_extra_retries" and value > 2):
                    raise ValueError(f"{name} is outside the supported retry range")
                continue
            if name in {"max_discovery_calls", "max_tool_schema_tokens"}:
                if type(value) is not int or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
                continue
            if name == "context_summary_trigger_fraction":
                if isinstance(value, bool) or not 0 < value <= 1:
                    raise ValueError(f"{name} must be between zero and one")
                continue
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive")
