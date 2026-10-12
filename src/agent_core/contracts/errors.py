"""Stable, failures for the independent agent boundary."""

from __future__ import annotations


class AgentContextError(ValueError):
    """A request or observation crossed its trusted agent boundary."""


class AgentInputBudgetError(AgentContextError):
    """The complete model view cannot fit without discarding required input."""

    code = "agent_input_budget_exceeded"

    def __init__(self, message: str, *, input_tokens: int, input_limit: int) -> None:
        self.input_tokens = input_tokens
        self.input_limit = input_limit
        self.public_message = (
            f"本次模型输入超过单次上下文额度（估算 {input_tokens}/{input_limit} token）。"
            "本轮尚未完成，需要缩小单次输入。"
        )
        super().__init__(message)


class AgentUpstreamError(RuntimeError):
    """Provider/data-source failure safe to expose through a stable code."""


class AgentTimeoutError(AgentUpstreamError):
    """A provider-neutral model timeout."""


class StructuredOutputError(AgentUpstreamError):
    """A bounded stage-output failure with no provider text in its public detail."""

    code = "structured_output_invalid"

    def __init__(self, *, truncated: bool = False, budget_exhausted: bool = False,
                 limit_message: str | None = None) -> None:
        self.public_message = (
            "模型输出达到长度上限，未能生成完整的阶段结果。"
            if truncated else "模型未能返回符合当前阶段格式的结果。"
        ) + ("本轮剩余额度不足，任务尚未完成。" if budget_exhausted else
             "已达到本阶段格式修复上限，任务尚未完成。")
        if limit_message:
            self.public_message += " " + limit_message
        super().__init__(self.public_message)


class ToolQueryTimeoutError(AgentUpstreamError):
    """A governed agent repository query exceeded its deadline."""


class AgentRecursionLimitError(AgentUpstreamError):
    """The graph unexpectedly exceeded its hard recursion boundary."""


class ToolCallingUnsupportedError(AgentUpstreamError):
    """The configured model cannot execute native structured tools."""


class OutputValidationError(AgentUpstreamError):
    """A final answer contains an unknown or malformed evidence citation."""


class AgentCancelled(RuntimeError):
    """Cooperative cancellation raised in planning, tools, or final output."""


class AgentError(RuntimeError):
    """Base failure with a stable code and a deliberately safe message."""

    code = "agent_error"
    public_message = "Agent查询暂时不可用"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.public_message)


class ToolNotRegisteredError(AgentError):
    code = "unknown_tool"
    public_message = "模型请求了未注册的工具"


class ToolPolicyDeniedError(AgentError):
    code = "tool_not_allowed"
    public_message = "当前上下文不允许执行该只读查询"


class ToolInputError(AgentError):
    code = "invalid_tool_arguments"
    public_message = "工具参数无效，请修正后重试"


class ToolExecutionError(AgentError):
    code = "tool_query_failed"
    public_message = "Agent查询失败"


class ObservationValidationError(AgentError):
    code = "unsafe_tool_observation"
    public_message = "查询结果未通过安全校验"


class AgentLimitReachedError(AgentError):
    code = "agent_limit_reached"
    public_message = "本轮自主查询已达到安全上限"

    def __init__(self, message: str | None = None) -> None:
        self.public_message = message or type(self).public_message
        super().__init__(self.public_message)


class DuplicateToolCallError(AgentError):
    code = "duplicate_tool_call"
    public_message = "重复的工具调用已被拒绝"


__all__ = [
    "AgentLimitReachedError",
    "DuplicateToolCallError",
    "ObservationValidationError",
    "ToolExecutionError",
    "ToolInputError",
    "ToolNotRegisteredError",
    "ToolPolicyDeniedError",
    "AgentError",
    "AgentRecursionLimitError",
    "AgentCancelled",
    "AgentContextError",
    "AgentInputBudgetError",
    "OutputValidationError",
    "ToolQueryTimeoutError",
    "AgentTimeoutError",
    "StructuredOutputError",
    "ToolCallingUnsupportedError",
    "AgentUpstreamError",
]


class ModelConfigurationError(RuntimeError):
    """Invalid model provider configuration."""
