"""Versioned, product-neutral plans and structured stage outputs."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from agent_core.contracts.errors import AgentContextError
from agent_core.contracts.runs import RunResult


PLAN_STATE_VERSION = "plan-execute-3"


class PlanActionError(AgentContextError):
    """The caller submitted an action against a different plan revision."""

    code = "plan_conflict"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class PlanInput(_Model):
    kind: Literal["message", "start_plan", "answer_questions", "execute_plan"] = "message"
    plan_id: str | None = None
    expected_revision: int | None = Field(default=None, ge=1)
    answers: dict[str, str] = Field(default_factory=dict)


class QuestionOption(_Model):
    label: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=600)


class PlanQuestion(_Model):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    title: str = Field(min_length=1, max_length=1000)
    options: list[QuestionOption] = Field(default_factory=list, max_length=5)


class StepSpec(_Model):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$")
    goal: str = Field(min_length=1, max_length=2000)
    expected_evidence: str = Field(min_length=1, max_length=2000)
    depends_on: list[str] = Field(default_factory=list, max_length=64)


class PlanStep(StepSpec):
    status: Literal["pending", "in_progress", "completed", "failed", "blocked"] = "pending"
    result: str = Field(default="", max_length=16000)
    unresolved: list[str] = Field(default_factory=list, max_length=32)
    source_ids: list[str] = Field(default_factory=list, max_length=128)


class PlanDraft(_Model):
    goal: str = Field(min_length=1, max_length=4000)
    scope: list[str] = Field(min_length=1, max_length=32)
    constraints: list[str] = Field(default_factory=list, max_length=32)
    acceptance: list[str] = Field(min_length=1, max_length=32)
    steps: list[StepSpec] = Field(min_length=1, max_length=64)

    @field_validator("scope", "constraints", "acceptance")
    @classmethod
    def nonblank_unique_items(cls, values):
        if any(not value.strip() for value in values) or len({value.strip() for value in values}) != len(values):
            raise ValueError("Scope, constraints and acceptance items must be nonblank and unique")
        return values


class ExplorationResult(_Model):
    """An ordinary Run artifact, never an implicit execution instruction."""

    kind: Literal["questions", "plan"]
    summary: str = Field(default="", max_length=4000)
    questions: list[PlanQuestion] = Field(default_factory=list, max_length=3)
    plan: PlanDraft | None = None
    unresolved: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_artifact(self):
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        if self.kind == "questions" and not self.questions:
            raise ValueError("questions artifact must contain a question")
        if self.kind == "plan" and (self.plan is None or self.questions or self.unresolved):
            raise ValueError("ready plan requires a plan and no unresolved questions")
        return self


class ExecutionStepResult(_Model):
    id: str = Field(min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_-]+$",
                    description="An exact ID from required_step_ids; do not renumber or substitute Todo IDs.")
    status: Literal["completed", "failed", "blocked"]
    result: str = Field(min_length=1, max_length=16000)
    source_ids: list[str] = Field(default_factory=list, max_length=128)
    unresolved: list[str] = Field(default_factory=list, max_length=32)


class ExecutionReport(_Model):
    """Archive the whole ReAct execution; this does not schedule individual steps."""

    steps: list[ExecutionStepResult] = Field(min_length=1, max_length=64,
        description="Exactly one entry for every required_step_id, including blocked or failed steps. Omit already completed steps.")
    new_conditions: list[str] = Field(default_factory=list, max_length=32)
    needs_replan: bool = False

    @model_validator(mode="after")
    def unique_steps(self):
        if len({step.id for step in self.steps}) != len(self.steps):
            raise ValueError("Execution report step IDs must be unique")
        return self


class ReplanDecision(_Model):
    kind: Literal["continue", "questions", "confirm_scope", "stop"]
    reason: str = Field(min_length=1, max_length=2000)
    remaining_steps: list[StepSpec] = Field(default_factory=list, max_length=64)
    questions: list[PlanQuestion] = Field(default_factory=list, max_length=3)
    proposed_plan: PlanDraft | None = None
    unresolved: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_decision(self):
        if len({question.id for question in self.questions}) != len(self.questions):
            raise ValueError("Question IDs must be unique")
        if self.kind == "questions" and not self.questions:
            raise ValueError("Questions decision must contain a question")
        if self.kind == "confirm_scope" and self.proposed_plan is None:
            raise ValueError("Scope confirmation requires a proposed plan")
        if self.kind == "continue" and (self.questions or self.proposed_plan):
            raise ValueError("Automatic replanning cannot change the approved scope")
        return self


class PlanState(_Model):
    version: str = PLAN_STATE_VERSION
    plan_id: str
    revision: int = Field(default=1, ge=1)
    phase: Literal["clarifying", "ready", "executing", "completed", "partial", "failed", "cancelled"] = "clarifying"
    goal: str = ""
    scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    acceptance: list[str] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    questions: list[PlanQuestion] = Field(default_factory=list)
    unresolved: list[str] = Field(default_factory=list)
    validation: dict[str, Any] = Field(default_factory=lambda: {"status": "not_run", "criteria": [], "iterations": 0,
                                                              "unresolved": [], "assessment": "model_assessment"})
    budget: dict[str, Any] = Field(default_factory=dict)
    revision_reason: str = ""
    revisions: list[dict[str, Any]] = Field(default_factory=list)
    approved_revision: int | None = None
    approved_scope: dict[str, Any] | None = None
    model_ledger: dict[str, Any] = Field(default_factory=dict)
    tool_totals: dict[str, Any] = Field(default_factory=dict)
    sources: dict[str, dict[str, Any]] = Field(default_factory=dict)
    last_run_id: str | None = None
    replans: int = 0
    active_step_id: str | None = None
    step_history: list[dict[str, Any]] = Field(default_factory=list)


def parse_plan(value: Any) -> PlanState | None:
    if value is None or value == {}:
        return None
    if isinstance(value, PlanState):
        plan = value.model_copy(deep=True)
    else:
        raw = value.get("plan_state") if isinstance(value, Mapping) and "plan_state" in value else value
        if not raw:
            return None
        plan = PlanState.model_validate(raw)
    if plan.version != PLAN_STATE_VERSION:
        raise PlanActionError("Saved plan version is incompatible")
    return plan


def migrate_plan_state(value: Any) -> dict[str, Any] | None:
    """Migrate a plan document for a new Run, never an old graph checkpoint."""
    if value is None or value == {}:
        return None
    raw = value.model_dump(mode="json") if isinstance(value, PlanState) else dict(value)
    if raw.get("version") == "plan-execute-1":
        raw = {**raw, "version": PLAN_STATE_VERSION, "active_step_id": None}
        raw["steps"] = [{**step, "status": "blocked",
                          "unresolved": list(dict.fromkeys([*step.get("unresolved", []), "旧版逐步执行已中断，需要继续核验"]))}
                         if step.get("status") == "in_progress" else step for step in raw.get("steps", [])]
    elif raw.get("version") == "plan-execute-2":
        raw = {**raw, "version": PLAN_STATE_VERSION}
    plan = parse_plan(raw)
    return plan.model_dump(mode="json") if plan is not None else None


def validate_plan_action(plan_state: Any, action: PlanInput | Mapping[str, Any]) -> None:
    action = action if isinstance(action, PlanInput) else PlanInput.model_validate(action)
    if action.kind != "answer_questions" and action.answers:
        raise PlanActionError("Answers are only accepted for the current question artifact")
    plan = parse_plan(plan_state)
    if plan is None:
        if action.kind not in {"message", "start_plan"} or action.plan_id is not None or action.expected_revision is not None:
            raise PlanActionError("The requested plan no longer exists")
        return
    if action.plan_id != plan.plan_id or action.expected_revision != plan.revision:
        raise PlanActionError("Plan changed; reload the current revision before continuing")
    if action.kind == "execute_plan":
        continuing = (plan.phase in {"failed", "partial"} and plan.approved_revision is not None
                      and plan.approved_scope == plan_scope(plan) and not plan.questions)
        if plan.phase != "ready" and not continuing:
            raise PlanActionError("Only a ready plan or an unfinished approved plan can execute")
    if action.kind == "answer_questions":
        expected = {question.id for question in plan.questions}
        if plan.phase != "clarifying" or not expected or set(action.answers) != expected:
            raise PlanActionError("Answers must address exactly the current questions")
        if any(not isinstance(answer, str) or not answer.strip() for answer in action.answers.values()):
            raise PlanActionError("Question answers cannot be empty")


def plan_scope(plan: PlanState | PlanDraft) -> dict[str, Any]:
    return {key: getattr(plan, key) for key in ("goal", "scope", "constraints", "acceptance")}


def validate_steps(steps: list[StepSpec], *, maximum: int) -> None:
    if not steps or len(steps) > maximum:
        raise ValueError(f"Plan must contain between 1 and {maximum} steps")
    seen: set[str] = set()
    for step in steps:
        if step.id in seen or len(step.depends_on) != len(set(step.depends_on)):
            raise ValueError("Plan step IDs and dependencies must be unique")
        if not set(step.depends_on) <= seen:
            raise ValueError("Serial plans may depend only on preceding steps")
        seen.add(step.id)


def project_plan(state_or_dict: Any) -> dict[str, Any] | None:
    plan = parse_plan(state_or_dict)
    if plan is None:
        return None
    return {
        "planId": plan.plan_id, "revision": plan.revision, "phase": plan.phase,
        "approvedRevision": plan.approved_revision,
        "goal": plan.goal, "scope": list(plan.scope), "constraints": list(plan.constraints),
        "acceptance": list(plan.acceptance),
        "steps": [{"id": step.id, "goal": step.goal, "expectedEvidence": step.expected_evidence,
                   "dependsOn": list(step.depends_on), "status": step.status, "result": step.result,
                   "unresolved": list(step.unresolved), "sourceIds": list(step.source_ids)} for step in plan.steps],
        "questions": [question.model_dump(mode="json") for question in plan.questions],
        "unresolved": list(plan.unresolved),
        "validation": {key: plan.validation[key] for key in
                       ("status", "criteria", "iterations", "unresolved", "assessment") if key in plan.validation},
        "budget": {key: plan.budget[key] for key in
                   ("modelCalls", "maxModelCalls", "inputTokens", "outputTokens", "totalTokens", "unknownUsageCalls",
                    "toolCalls", "maxToolCalls", "observationBytes", "maxObservationBytes", "externalToolCalls",
                    "maxExternalToolCalls", "memoryToolCalls", "maxMemoryToolCalls", "stopReason", "limitMessage",
                    "replans", "maxReplans") if key in plan.budget} | ({"toolLimits": {
                        name: {key: counts[key] for key in ("used", "limit", "remaining") if key in counts}
                        for name, counts in plan.budget["toolLimits"].items() if isinstance(counts, Mapping)
                    }} if isinstance(plan.budget.get("toolLimits"), Mapping) else {}),
        "revisionReason": plan.revision_reason,
        "revisions": [{"revision": item["revision"], "reason": item["reason"]} for item in plan.revisions],
        "stepHistory": [{key: item[key] for key in ("stepId", "status", "result", "unresolved", "sourceIds", "revision", "runId")
                         if key in item} for item in plan.step_history],
    }


def _preview_line(value: str, maximum: int = 160) -> str:
    """Keep model-authored fields on one bounded Markdown line."""
    line = " ".join(value.split())
    return line if len(line) <= maximum else line[:maximum - 1].rstrip() + "…"


def _preview_items(lines: list[str], title: str, items: list[str], *, limit: int = 4) -> None:
    if not items:
        return
    lines.extend((f"**{title}**", *[f"- {_preview_line(item)}" for item in items[:limit]]))
    if len(items) > limit:
        lines.append(f"- 其余 {len(items) - limit} 项见结构化计划")
    lines.append("")


def render_plan_control_text(plan: PlanState) -> str:
    """Summarize a pending plan for the same guarded answer channel as other replies."""
    if plan.questions:
        lines = [f"### 待补充信息（计划版本 {plan.revision}）", ""]
        for index, question in enumerate(plan.questions, 1):
            lines.append(f"{index}. {_preview_line(question.title, 240)}")
            for option in question.options:
                label = _preview_line(option.label, 100)
                description = _preview_line(option.description, 120)
                lines.append(f"   - {label}" + (f"：{description}" if description else ""))
        lines.extend(("", "回答问题后会更新计划，不会自动执行。"))
        return "\n".join(lines)
    if plan.phase != "ready":
        raise ValueError("Only a ready plan or current questions have a control preview")

    lines = [f"### 计划预览（版本 {plan.revision}，待确认）", "", f"**目标**：{_preview_line(plan.goal, 240)}", ""]
    _preview_items(lines, "范围", plan.scope)
    _preview_items(lines, "约束", plan.constraints)
    lines.append("**步骤与预期证据**")
    for index, step in enumerate(plan.steps[:8], 1):
        status = "（已完成）" if step.status == "completed" else ""
        lines.append(f"{index}. `{step.id}`{status} {_preview_line(step.goal, 160)}；证据：{_preview_line(step.expected_evidence, 120)}")
    if len(plan.steps) > 8:
        lines.append(f"其余 {len(plan.steps) - 8} 步见结构化计划。")
    lines.append("")
    _preview_items(lines, "验收标准", plan.acceptance)
    lines.append("请审阅后选择执行，或继续补充要求；完整计划以结构化计划为准。")
    return "\n".join(lines)


def plan_result(state: Mapping[str, Any]) -> RunResult:
    plan = project_plan(state)
    kind = "questions" if plan and plan["questions"] else "plan_preview" if plan and plan["phase"] == "ready" else "final_result"
    return RunResult(output=state.get("final_answer", ""), artifacts=tuple(state.get("references", ())),
                     metadata={"plan": plan, "artifactKind": kind})


__all__ = ["PLAN_STATE_VERSION", "PlanActionError", "PlanInput", "PlanState", "PlanDraft", "PlanStep",
           "StepSpec", "ExecutionStepResult", "ExecutionReport", "PlanQuestion", "QuestionOption", "ExplorationResult", "ReplanDecision",
           "project_plan", "plan_result", "validate_plan_action", "migrate_plan_state"]
