"""Run context and application policies for the reusable ReAct graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_core.config import AgentConfig
from agent_core.contracts.errors import AgentContextError, OutputValidationError
from agent_core.contracts.runs import RunRequest


class CitationStreamGuard(Protocol):
    def feed(self, text: str) -> Sequence[str]: ...

    def finish(self) -> Sequence[str]: ...


class SensitiveStreamGuard(Protocol):
    def feed(self, text: str) -> str: ...

    def finish(self) -> str: ...


class _PlainTextGuard:
    def feed(self, text: str) -> Sequence[str]:
        return (text,) if text else ()

    def finish(self) -> Sequence[str]:
        return ()


class _SensitiveTextGuard:
    """Hold partial sensitive strings before releasing streamed output."""

    def __init__(self, values: Sequence[str]) -> None:
        self.values = tuple(sorted({value for value in values if value}, key=len, reverse=True))
        self.pending = ""
        self.maximum = max((len(value) for value in self.values), default=0)

    def feed(self, text: str) -> str:
        if not self.values:
            return text
        value = self.pending + text
        self.pending = ""
        if any(secret in value for secret in self.values):
            raise OutputValidationError("最终回答包含不可公开的工具数据")
        upper = min(len(value), max(0, self.maximum - 1))
        for length in range(upper, 0, -1):
            suffix = value[-length:]
            if any(secret.startswith(suffix) for secret in self.values):
                self.pending = suffix
                return value[:-length]
        return value

    def finish(self) -> str:
        if len(self.pending) >= 3:
            raise OutputValidationError("最终回答包含不完整的敏感工具数据")
        value = self.pending
        self.pending = ""
        return value


class ReActPolicy:
    """Application hooks; the default supports ordinary, uncited tools.

    Applications may supply prompts, contextual validation, output rules and
    display wording without importing their product state into the graph.
    """

    TEXT: Mapping[str, str] = {
        "invalid_tool_call": "工具名称或参数不符合已注册的工具契约",
        "tool_calling_unsupported": "模型不支持工具调用",
        "missing_tool_call_id": "模型工具调用缺少标识",
        "duplicate_tool_call_id": "模型返回了重复工具调用标识",
        "answer_too_long": "模型回答超过安全长度限制",
        "empty_model_result": "模型返回空结果",
        "empty_public_answer": "本轮没有生成可用结果。",
        "sensitive_output": "最终回答包含不可公开的工具数据",
        "context_step": "校验可信上下文",
        "final_step": "生成有依据回答",
        "validation_step": "校验回答",
    }

    def text(self, key: str, **values: Any) -> str:
        return self.TEXT[key].format(**values)

    def validate_context(self, context: ReActContext) -> None:
        if not context.request.run_id or not context.request.workflow_id:
            raise AgentContextError("运行缺少标识或流程")

    def planner_prompt(self, context: ReActContext, tool_names: Sequence[str]) -> str:
        return (
            "根据用户请求和已有工具结果，决定是否需要调用已注册工具。"
            "只在确有必要时调用工具；可以回答时不再调用工具。"
            "需要工具时只发起工具调用；可以回答时直接生成最终回答，不要输出内部思维链。"
            f"\n可用工具：{', '.join(tool_names)}"
        )

    def required_direct_tools(self, context: ReActContext, state: Mapping[str, Any] | None = None) -> Sequence[str]:
        """Host-owned obligations, such as an explicit memory save request."""
        return ()

    def optional_direct_tools(self, context: ReActContext) -> Sequence[str]:
        return ()

    def final_prompt(self, state: Mapping[str, Any]) -> str:
        return (
            "根据用户请求、对话和已有工具结果生成最终回答。"
            "不要编造工具结果，不要声称执行了未发生的动作；明确必要的不确定性。"
        )

    def structured_prompt(
        self, context: ReActContext, tool_names: Sequence[str], output_tool_names: Sequence[str],
    ) -> str:
        """Stage guidance without the ordinary final-answer stop instruction."""
        return (
            "根据用户目标和当前阶段要求使用必要工具，返回简洁的结构化阶段产物。"
            "只报告实际获得的证据，不编造结果或声称执行未发生的动作。"
            "证据不足时在约定字段中说明缺项，不输出内部思维链。"
        )

    def citation_guard(self, references: Sequence[Any]) -> CitationStreamGuard:
        return _PlainTextGuard()

    def sensitive_guard(self, values: Sequence[str]) -> SensitiveStreamGuard:
        return _SensitiveTextGuard(values)

    def validate_output(self, answer: str, references: Sequence[Any]) -> None:
        """No citation syntax or domain-specific output is required by default."""

    def control_output_policy(self, state: Mapping[str, Any]) -> ReActPolicy:
        """Policy for questions, previews or incomplete progress summaries.

        Hosts may relax answer-specific evidence requirements while retaining
        their guards for sensitive data and any references actually displayed.
        """
        return self

    def limit_notice(self, stop_reason: str | None) -> str:
        if stop_reason in {
            "planning_limit", "tool_call_limit", "per_tool_limit",
            "observation_limit", "reference_limit", "category_tool_limit",
        }:
            return "已达到本轮执行上限，以下回答基于已获得的信息。"
        return ""

    def validation_payload(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return {}

    def after_validation_events(self, state: Mapping[str, Any]) -> Sequence[dict[str, Any]]:
        return ()


@dataclass(frozen=True)
class ReActContext:
    request: RunRequest = field(repr=False)
    cancellation: Any
    settings: AgentConfig
    async_stream_runner: Any = field(repr=False)
    clock: Callable[[], Any] = field(repr=False)
    tool_runtime: Any = field(repr=False)
    policy: ReActPolicy
    memory: Any = field(default=None, repr=False)
    model_budget: Any = field(default=None, repr=False)


__all__ = ["ReActContext", "ReActPolicy"]
