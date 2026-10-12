"""A small serial workflow around official ReAct and rubric components.

The graph owns plan transitions; create_agent owns every model/tool loop.
Human questions and previews finish a Run normally. Only an explicit action
against the current ready revision starts work in a later Run.
"""

from __future__ import annotations

from agent_core.runtime.streaming import call_async, run_sync
from agent_core.tools.contracts import AuthorizationDecision

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from deepagents.middleware.rubric import RubricMiddleware
from langchain.agents.middleware import AgentMiddleware, ModelResponse, TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Overwrite
from pydantic import BaseModel, ValidationError

from agent_core.contracts.errors import AgentCancelled, AgentContextError, AgentInputBudgetError, AgentLimitReachedError, AgentUpstreamError, OutputValidationError, StructuredOutputError, ToolPolicyDeniedError
from agent_core.models.langchain import StreamingChatModelAdapter
from agent_core.memory.binding import memory_binding_checkpoint, restore_memory_binding
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.middleware import ReActModelMiddleware
from agent_core.planning.react.output import publish_answer
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.planning.react.recovery import aprepare_react_resume
from agent_core.runtime.budget import ExecutionBudget
from agent_core.tools.contracts import ToolDiscoverySpec

from .budget import PlanModelBudget
from .contracts import (
    ExplorationResult, ExecutionReport, PlanActionError, PlanDraft, PlanInput, PlanState, PlanStep, ReplanDecision,
    parse_plan, plan_scope, project_plan, render_plan_control_text, validate_plan_action, validate_steps,
)
from .state import PlanExecuteState


_READ_MEMORY_TOOLS = frozenset({"ls", "read_file", "glob", "grep", "history_read", "history_search", "memory_read", "memory_list", "write_todos"})
_REPORT_CONTEXT_TOOLS = _READ_MEMORY_TOOLS - {"write_todos"}
_MEMORY_KEYS = ("files", "todos", "task_source_ids", "loaded_memory_releases", "memory_notes_hash",
                "memory_release_snapshot", "memory_contents", "skills_metadata", "skills_load_errors",
                "run_exposed_memory_sources", "run_memory_exposure_complete", "context_summary", "summary_calls",
                "context_warning", "context_history_dependencies", "context_has_legacy_history")
_CONTEXT_PREPARATION_KEYS = frozenset({"messages", "context_summary", "summary_calls", "context_warning",
                                       "context_history_dependencies", "context_has_legacy_history"})
_STAGE_EVENT_KINDS = frozenset({"step.state", "tool.planned", "tool.started", "tool.completed",
                              "reference.added", "context.selected", "memory.saved", "memory.proposed", "memory.updated"})
_STAGE_INSTRUCTIONS = {
    "explore": """你正在形成可审阅的计划，尚未获得执行计划的批准。只进行必要的只读探索。
需要会改变结果的用户信息时输出 questions（一次最多3题，给出少量清晰选项）；不要猜测答案。
信息充分时输出 plan：目标、明确范围、约束、逐项验收标准、串行步骤及每步预期证据。
不要执行计划，不要把待办完成或规划文字当执行结果。保留已完成步骤；计划最多包含配置给出的步数。
只返回 ExplorationResult 结构化产物，不要输出内部思维链。""",
    "execute": """用户已经明确确认以下整份计划。使用现有ReAct完成整份计划的剩余工作，复用已取得的证据。
根据实际结果自主决定工具调用与先后次序，Todo记录内部进度；不需要逐步提交结构化结果。
已完成步骤及其事实不得重复执行或改写，不扩大批准的目标、范围、约束或验收口径。
遇到关键未知、工具限制或需改变范围时停止相关工作，在普通文字答复中准确说明已完成内容、证据与缺项。
调用结束后再统一归档和验收；本阶段不能自行宣布任务通过验收。""",
    "report": """依据消息中的整计划ReAct答复和真实工具观察，统一归档执行结果；不得再次进行业务查询或写入。
系统中的来源目录只提供定位摘要，excerptTruncated明确标记省略，不能把摘要当完整证据。
需要完整正文时使用已有history_read/history_search或只读文件工具回查；材料不可取得或仍不充分时必须blocked。
返回ExecutionReport，steps必须恰好覆盖计划中尚未标记completed的全部步骤，逐项包含id、状态、结果与未解决问题。
使用required_step_ids给出的原始ID，不能用Todo序号、步骤标题或重新编号替代；失败和受阻步骤也必须逐项报告。
已完成步骤由控制器保留，不得重新提交或改写。source_ids只能使用控制器列出的真实来源ID，不能编造。
每个completed步骤必须有完整结果、可回查来源、无未解决问题，并满足其依赖；缺证据或结果不确定应blocked。
需要范围内调整剩余计划时标记needs_replan或new_conditions；这份报告只是归档，不能代替最终验收。""",
    "replan": """根据真实步骤结果调整剩余计划。已经完成的步骤、已批准的目标/范围/约束/验收均不可改写。
continue只返回剩余步骤；保留确实有必要的失败步骤并改进做法，避免重复已完成工作。
需要关键用户信息时输出questions；确实需要改变批准范围时输出confirm_scope和新的可审阅计划。
无法在剩余额度内完成时输出stop并明确未解决问题。不能把缺证据解释为已完成。返回ReplanDecision。""",
    "final": """所有计划步骤已完成并经过确定性检查。依据批准计划、每步结果和真实来源生成最终答复。
逐项覆盖验收标准，交代证据及限制；模型复核只是模型评价，不能宣称独立事实验证。
复核提出文字遗漏时据已有证据修改答复，不得虚构新工具观察或额外完成的步骤。""",
}


def _tool_modes(context: Any) -> dict[str, bool]:
    return {"discovery": context.settings.tool_discovery_enabled, "retry": context.settings.tool_retry_enabled}


def _validate_tool_modes(state: Mapping[str, Any], context: Any) -> None:
    if state.get("tool_modes") is not None and state["tool_modes"] != _tool_modes(context):
        raise AgentContextError("Cannot change tool discovery/retry mode while resuming a Run")


def _budget(context: Any) -> PlanModelBudget:
    budget = getattr(context, "model_budget", None)
    if not isinstance(budget, PlanModelBudget):
        raise AgentContextError("Plan-and-Execute requires a shared PlanModelBudget in its context")
    return budget


def _validate_public_strings(value: Any, policy: ReActPolicy, sensitive_values: Sequence[str]) -> None:
    """Apply the output guard to every public structured string, before commit."""
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, str):
        guard = policy.sensitive_guard(sensitive_values)
        guard.feed(value)
        guard.finish()
    elif isinstance(value, Mapping):
        for item in value.values():
            _validate_public_strings(item, policy, sensitive_values)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_strings(item, policy, sensitive_values)


class _StageMiddleware(AgentMiddleware):
    state_schema = PlanExecuteState

    def __init__(self, stage: str, allowed_tools: set[str], *, max_steps: int) -> None:
        self.stage, self.allowed_tools, self.max_steps = stage, allowed_tools | {"search_tools"}, max_steps

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        plan = parse_plan(request.state.get("plan_state"))
        sources = plan.sources if plan else {}
        excerpt_limit = min(600, 6000 // max(1, len(sources)))
        catalog = {identity: {
            "kind": value.get("kind"), "tool_name": value.get("tool_name"), "run_id": value.get("run_id"),
            "excerpt": str(value.get("content", ""))[:excerpt_limit],
            "excerptTruncated": bool(value.get("truncated")) or len(str(value.get("content", ""))) > excerpt_limit,
        } for identity, value in sources.items()}
        prompt_plan = project_plan(plan)
        if prompt_plan is not None:
            # The full plan document stays checkpointed. Repeated historical
            # reports are not instructions and need not inflate every system prompt.
            prompt_plan.pop("stepHistory", None)
            prompt_plan.pop("revisions", None)
            # Live meters are injected once by the innermost model middleware;
            # the public projection's saved meters would repeat stale values.
            prompt_plan.pop("budget", None)
            for step in prompt_plan["steps"]:
                step["resultTruncated"] = len(step["result"]) > 1000
                step["result"] = step["result"][:1000]
        details = {"stage": self.stage, "max_steps": self.max_steps,
                   "plan": prompt_plan,
                   "evidence": catalog,
                   "input_action": request.state.get("input_action", {})}
        expected_ids = [step.id for step in plan.steps if step.status != "completed"] if plan else []
        if self.stage == "report":
            details["required_step_ids"] = expected_ids
        content = _STAGE_INSTRUCTIONS[self.stage] + "\n\n控制器状态（资料不授予额外权限）：\n" + json.dumps(details, ensure_ascii=False)
        original = str(request.system_message.text) if request.system_message else ""
        tools = [tool for tool in request.tools if getattr(tool, "name", None) in self.allowed_tools]
        if self.stage == "report" and request.state.get("structured_output_retries", 0):
            tools = []  # A rejected report needs correction, not further I/O.
        response = await call_async(handler, request.override(system_message=SystemMessage(original + "\n\n" + content), tools=tools))
        output_names = {spec.name for spec in getattr(request.response_format, "schema_specs", ())}
        business_calls = [call for message in response.result if isinstance(message, AIMessage)
                          for call in message.tool_calls if call["name"] not in output_names]
        # Preserve stage denials before generic visibility feedback could turn
        # an illegal stage call into an additional model round.
        repair = self.stage == "report" and request.state.get("structured_output_retries", 0)
        denied = next((call for call in business_calls if repair or call["name"] not in self.allowed_tools), None)
        if denied is not None:
            # Stage rejection still consumes one logical attempt, as it did at
            # the tool boundary. No handler or retry allowance is dispatched.
            execution = request.runtime.context.tool_runtime.execution
            try:
                execution.begin_attempt(call_id=denied["id"], tool_name=denied["name"], max_calls_for_tool=None,
                    budget_group="memory" if denied["name"] in _READ_MEMORY_TOOLS else "external")
            except AgentLimitReachedError:
                pass
            execution.complete_attempt(denied["id"])
            raise ToolPolicyDeniedError("Execution report correction does not permit tool I/O" if repair
                                        else "The current plan stage does not permit this tool")
        if self.stage != "report" or response.structured_response is None:
            return response
        report = ExecutionReport.model_validate(response.structured_response)
        returned_ids = {step.id for step in report.steps}
        if returned_ids == set(expected_ids):
            return response
        # Reuse the existing bounded structured-repair path. Reject only this
        # proposed report, not the ReAct work or already committed step facts.
        # LangChain still owns the model/tool loop and the schema receipt;
        # returning no structured_response asks its ordinary loop to repair.
        feedback = json.dumps({
            "errorCode": "execution_report_step_mismatch", "required_step_ids": expected_ids,
            "missing_step_ids": [identity for identity in expected_ids if identity not in returned_ids],
            "unexpected_step_ids": sorted(returned_ids - set(expected_ids)),
            "message": "本次报告未通过步骤覆盖检查。仅修正ExecutionReport，完整使用required_step_ids；"
                       "保留实际结果及来源，无法完成的步骤填写blocked及缺项，不要重新执行业务查询或降低验收条件。",
        }, ensure_ascii=False)
        return ModelResponse(result=[message.model_copy(update={"content": feedback})
                                     if isinstance(message, ToolMessage) and message.name == "ExecutionReport"
                                     else message for message in response.result], structured_response=None)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        if self.stage == "report" and request.state.get("structured_output_retries", 0):
            raise ToolPolicyDeniedError("Execution report correction does not permit tool I/O")
        if request.tool_call["name"] not in self.allowed_tools:
            raise ToolPolicyDeniedError("The current plan stage does not permit this tool")
        return await call_async(handler, request)

    def wrap_model_call(self, request, handler):
        return run_sync(self.awrap_model_call(request, handler))

    def wrap_tool_call(self, request, handler):
        return run_sync(self.awrap_tool_call(request, handler))


class _RubricOutcomeMiddleware(AgentMiddleware):
    """Project official private rubric bookkeeping into our owned outcome."""

    state_schema = PlanExecuteState

    def after_agent(self, state: Any, runtime: Runtime) -> dict[str, Any]:
        del runtime
        return {"plan_rubric": {"status": state.get("_rubric_status"),
                               "iterations": state.get("_rubric_iterations", 0),
                               "evaluations": state.get("_rubric_evaluations", [])}}


class _CandidateValidationMiddleware(AgentMiddleware):
    """Check every proposed final answer before the official rubric runs."""

    def __init__(self, policy: ReActPolicy) -> None:
        self.policy = policy

    def after_model(self, state: Any, runtime: Runtime) -> None:
        candidate = next((message for message in reversed(state["messages"]) if isinstance(message, AIMessage)), None)
        if candidate is None or candidate.tool_calls:
            raise OutputValidationError("Final review did not produce an answer candidate")
        execution = runtime.context.tool_runtime.execution
        text = str(candidate.text)
        if not text.strip() or candidate.invalid_tool_calls:
            raise OutputValidationError("Final review did not produce a valid nonblank answer candidate")
        guard = self.policy.sensitive_guard(tuple(execution.sensitive_values))
        guard.feed(text)
        guard.finish()
        self.policy.validate_output(text, execution.references_snapshot())


class _PlanNativeAuthorizer:
    def __init__(self, owned_names, host):
        self.owned_names, self.host = frozenset(owned_names), host

    async def authorize(self, **kwargs):
        if kwargs["tool_name"] in self.owned_names:
            return AuthorizationDecision(True)
        if self.host is not None:
            return await call_async(getattr(self.host, "authorize", self.host), **kwargs)
        return AuthorizationDecision(False, "tool_not_allowed", "Native tool requires explicit authorization")


class _PlanWorkflow:
    def __init__(self, model: Any, tools: Sequence[BaseTool], *, exploration_tools: Sequence[BaseTool],
                 stream_chat_message: Callable[..., Any], policy: ReActPolicy, context_schema: type[Any],
                 middleware: Sequence[AgentMiddleware], memory: Any, store: Any, action: PlanInput | None,
                 max_steps: int, max_replans: int, max_rubric_iterations: int, max_model_calls: int | None,
                 checkpointer: Any, context_summarizer: Any = None, prepare_context: Callable[..., Any] | None = None, native_authorizer: Any = None) -> None:
        self.action, self.policy = action, policy
        self.context_summarizer, self.context_preparer = context_summarizer, prepare_context
        self.max_steps, self.max_replans = max_steps, max_replans
        self.max_rubric_iterations, self.max_model_calls = max_rubric_iterations, max_model_calls
        self.persisted = checkpointer is not None
        self.active_child: tuple[Any, dict[str, Any]] | None = None
        names = {tool.name for tool in tools}
        explore_names = {tool.name for tool in exploration_tools}
        if not explore_names <= names:
            raise ValueError("Exploration tools must be registered execution tools")
        self.graphs = {}
        for stage, schema, allowed in (
            ("explore", ExplorationResult, explore_names | _READ_MEMORY_TOOLS),
            ("execute", None, names | _READ_MEMORY_TOOLS | {"write_file", "edit_file", "memory_remember", "memory_correct", "memory_forget", "memory_report_used", "skill_propose", "workflow_propose"}),
            ("report", ExecutionReport, _REPORT_CONTEXT_TOOLS),
            ("replan", ReplanDecision, _READ_MEMORY_TOOLS),
        ):
            extra = [] if memory is not None or stage == "report" else [TodoListMiddleware()]
            for component in extra:
                for tool in component.tools:
                    tool.extras = {**(tool.extras or {}), "agent_core_catalog": {
                        "version": "1.0.0", "source": "plan-components", "category": "planning",
                        "budget_group": "memory", "effect": "control", "allowed_state_keys": ("messages", "todos"),
                        "discovery": ToolDiscoverySpec(aliases=("更新待办", "任务计划")),
                    }}
            self.graphs[stage] = compile_react_graph(
                model, tools, context_schema=context_schema, stream_chat_message=stream_chat_message, policy=policy,
                middleware=[*middleware, *extra, _StageMiddleware(stage, set(allowed), max_steps=max_steps)],
                memory=memory, store=store, checkpointer=checkpointer,
                publish_final=False, response_format=ToolStrategy(schema) if schema is not None else None,
                context_summarizer=context_summarizer,
                native_authorizer=(_PlanNativeAuthorizer({tool.name for component in extra for tool in component.tools}, native_authorizer)
                                   if extra else native_authorizer),
            )
        # The official rubric owns its bounded answer/revision loop. The grader
        # has no business tools and must not archive its synthetic Human payload
        # as the current user's original message.
        grader_model = StreamingChatModelAdapter(delegate=model, collect=stream_chat_message)
        rubric = RubricMiddleware(
            model=grader_model, max_iterations=max_rubric_iterations,
            grader_middleware=[ReActModelMiddleware(policy, stream_chat_message, archive_history=False)],
            grader_context_schema=context_schema,
        )
        self.graphs["final"] = compile_react_graph(
            model, (), context_schema=context_schema, stream_chat_message=stream_chat_message, policy=policy,
            middleware=[*middleware, _StageMiddleware("final", set(), max_steps=max_steps),
                        _CandidateValidationMiddleware(policy), _RubricOutcomeMiddleware(), rubric],
            memory=None, store=store, checkpointer=checkpointer, publish_final=False,
            context_summarizer=context_summarizer,
        )

    def _save(self, plan: PlanState, context: Any) -> dict[str, Any]:
        model = _budget(context)
        execution = context.tool_runtime.execution
        plan.model_ledger = model.snapshot()
        ledger = execution.snapshot()
        plan.tool_totals = {key: deepcopy(ledger[key]) for key in
                            ("tool_call_count", "observation_bytes", "calls_by_tool", "calls_by_category")}
        plan.budget = {**model.public(), **execution.public_budget(),
                       "replans": plan.replans, "maxReplans": self.max_replans}
        return {"plan_state": plan.model_dump(mode="json"), "model_ledger": model.snapshot(),
                "tool_modes": _tool_modes(context),
                "execution_ledger": ledger, "tool_call_count": execution.tool_call_count}

    def _restore(self, state: Mapping[str, Any], context: Any) -> PlanState:
        _validate_tool_modes(state, context)
        plan = parse_plan(state["plan_state"])
        assert plan is not None
        _budget(context).restore(plan.model_ledger)
        _budget(context).restore(state.get("model_ledger", {}))
        context.tool_runtime.execution.restore(state.get("execution_ledger", {}))
        return plan

    @staticmethod
    def _revision(plan: PlanState, reason: str) -> None:
        plan.revision += 1
        plan.revision_reason = reason
        plan.revisions.append({"revision": plan.revision, "reason": reason})

    @staticmethod
    def _sources(plan: PlanState, messages: Sequence[Any], context: Any) -> list[str]:
        added = []
        for message in messages:
            is_tool = isinstance(message, ToolMessage) and message.tool_call_id in context.tool_runtime.execution.completed_call_ids
            is_user = isinstance(message, HumanMessage) and not message.additional_kwargs.get("lc_source")
            if not (is_tool or is_user):
                continue
            if is_tool and message.status == "error":
                continue
            identity = message.id or ("tool-" + hashlib.sha256(message.tool_call_id.encode()).hexdigest()[:32] if is_tool else "user_message")
            source_id = message.additional_kwargs.get("history_source_id") or f"{context.request.run_id}:{identity}"
            source_run_id = message.additional_kwargs.get("history_run_id") or context.request.run_id
            if source_id not in plan.sources:
                text = str(message.text)
                for private in sorted(context.tool_runtime.execution.sensitive_values, key=len, reverse=True):
                    if private:
                        text = text.replace(private, "[redacted]")
                plan.sources[source_id] = {"kind": "tool" if is_tool else "user", "run_id": source_run_id,
                                           "content": text[:12000], "truncated": len(text) > 12000,
                                           "tool_name": message.name if is_tool else None,
                                           "step_id": plan.active_step_id if is_tool else None}
                added.append(source_id)
        return added

    async def prepare(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        del config
        context = runtime.context
        _validate_tool_modes(state, context)
        context.cancellation.raise_if_cancelled()
        model = _budget(context)
        limits = [limit for limit in (model.max_calls, self.max_model_calls) if limit is not None]
        model.max_calls = min(limits) if limits else None
        action = self.action or PlanInput.model_validate(state.get("input_action") or {})
        plan = parse_plan(state.get("plan_state"))
        validate_plan_action(plan, action)
        new_window = plan is None or action.kind == "start_plan"
        reset_updates = {}
        if action.kind == "start_plan" and plan is not None and plan.last_run_id == context.request.run_id:
            raise PlanActionError("Starting a new plan requires a new Run; resume this Run's saved progress")
        if new_window:
            # Explicitly starting a different plan is the sole reset boundary.
            # Keep the task's Notes/History context, never the prior plan's
            # approval, execution facts, citations or consumed budget window.
            model_revision = max(model.revision, state.get("model_ledger", {}).get("revision", 0),
                                 plan.model_ledger.get("revision", 0) if plan else 0)
            model.restore({**PlanModelBudget(max_calls=model.max_calls).snapshot(), "revision": model_revision + 1})
            execution = context.tool_runtime.execution
            execution.restore({**ExecutionBudget().snapshot(), "limits": {},
                               "revision": max(execution.revision, state.get("execution_ledger", {}).get("revision", 0)) + 1})
            reset_updates = {"references": Overwrite([]), "observation_bytes": Overwrite(0),
                             "used_tool_call_ids": [], "planning_rounds": 0, "structured_output_retries": 0,
                             "todos": [], "plan_rubric": {}, "execution_answer": "", "stage_output": {},
                             "context_summary": {}, "summary_calls": 0, "context_warning": ""}
            identity = hashlib.sha256(context.request.run_id.encode()).hexdigest()[:24]
            plan = PlanState(plan_id=f"plan-{identity}", goal=context.request.input,
                             revision_reason="首次规划")
        model.restore(plan.model_ledger)
        memory_updates = {}
        if context.memory is not None:
            await call_async(getattr(context.memory, "arefresh", context.memory.refresh))
            memory_updates["memory_release_snapshot"] = memory_binding_checkpoint(context.memory)
        execution = context.tool_runtime.execution
        if not new_window:
            if plan.last_run_id != context.request.run_id and plan.tool_totals:
                # Carry cumulative allowances, never another Run's call identities,
                # private values or incomplete tool cache. Safe Evidence objects
                # belong to the whole plan and retain their original numbering.
                execution.restore({"revision": execution.revision + 1, **plan.tool_totals,
                                   "references": list(state.get("references", [])), "used_call_ids": [], "completed_call_ids": [],
                                   "sensitive_values": [], "stop_reason": None})
            else:
                execution.restore(state.get("execution_ledger", {}))
        plan.last_run_id = context.request.run_id
        self._sources(plan, state.get("messages", []), context)
        if action.kind == "execute_plan":
            if plan.questions or (plan.phase == "ready" and plan.unresolved):
                raise AgentContextError("A plan with unresolved questions cannot execute")
            validate_steps(plan.steps, maximum=self.max_steps)
            plan.approved_revision = plan.revision
            plan.approved_scope = deepcopy(plan_scope(plan))
            plan.phase = "executing"
            plan.validation = {"status": "not_run", "criteria": [], "iterations": 0,
                               "unresolved": [], "assessment": "model_assessment"}
            plan.active_step_id = None
            route = "execute"
        else:
            plan.phase = "clarifying"
            plan.approved_revision = None
            plan.approved_scope = None
            route = "explore"
        return {**self._save(plan, context), **memory_updates, "input_action": action.model_dump(mode="json"),
                "route": "prepare_context" if self.context_preparer is not None else route,
                "stage_failure": None, "final_answer": "",
                "observation_bytes": max(0, execution.observation_bytes - state.get("observation_bytes", 0)),
                **reset_updates}

    async def prepare_context(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        """Prepare history only after this plan's budget window is checkpointed."""
        del config
        context = runtime.context
        plan = self._restore(state, context)
        context.cancellation.raise_if_cancelled()
        updates = dict(await call_async(self.context_preparer, state, context) or {})
        if updates.keys() - _CONTEXT_PREPARATION_KEYS:
            raise AgentContextError("Plan context preparation may only update messages and derived context")
        if "messages" in updates:
            # Use the very same identity in plan provenance, History and the
            # graph channel; add_messages otherwise assigns it only after this
            # node returns. Existing source identities remain unchanged.
            updates["messages"] = [message if message.id else message.model_copy(update={"id":
                "prepared-" + hashlib.sha256(f"{context.request.run_id}:{index}".encode()).hexdigest()[:32]})
                for index, message in enumerate(updates["messages"])]
        self._sources(plan, updates.get("messages", state.get("messages", [])), context)
        return {**updates, **self._save(plan, context),
                "route": "execute" if state["input_action"]["kind"] == "execute_plan" else "explore"}

    async def _invoke(self, name: str, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> tuple[dict[str, Any], Any, list[str]]:
        context = runtime.context
        plan = self._restore(state, context)
        context.cancellation.raise_if_cancelled()
        execution = context.tool_runtime.execution
        previous = {message.id for message in state.get("messages", [])}
        initial = {**dict(state), "plan_stage": name, "planning_rounds": 0,
                   "activity_scope_id": self._activity_scope(plan, name),
                   "activity_label": {"explore": "探索与规划", "execute": "计划执行", "report": "执行结果核对",
                                      "replan": "重规划", "final": "最终验收"}[name],
                   "tool_call_count": execution.tool_call_count, "stop_reason": None,
                   "structured_response": None, "final_answer": "", "used_tool_call_ids": sorted(execution.used_call_ids),
                   "execution_ledger": execution.snapshot(), "model_ledger": _budget(context).snapshot(),
                   "references": list(execution.references_snapshot()), "observation_bytes": execution.observation_bytes}
        if name == "final":
            initial["rubric"] = "逐项按以下冻结验收标准复核答复。每条name必须原样回传完整的 A编号、冒号及标准正文，不可改写、合并、遗漏或发明标准。\n" + "\n".join(
                f"A{index}: {criterion}" for index, criterion in enumerate(plan.acceptance, 1))
        # A stable namespace survives a root terminal-budget checkpoint update,
        # which may give the re-entered parent node a new task UUID. Thread IDs
        # remain Run-owned; plan versions, stages and replans distinguish invocations.
        child_config = {**config, "configurable": {**config.get("configurable", {}),
            "checkpoint_ns": f"plan_stage:{plan.plan_id}_{plan.revision}_{name}_{plan.active_step_id or 'none'}_{plan.replans}"}}
        child_config["configurable"].pop("checkpoint_id", None)
        child_values: dict[str, Any] = {}

        async def failure_update(reason: str) -> dict[str, Any]:
            # Tool observations can commit before the stage's final model call
            # fails. Retain their public evidence and consumed observation bytes.
            values = child_values
            if self.persisted and self.active_child is not None:
                child, saved_config = self.active_child
                snapshot = await child.with_config({"configurable": {"checkpoint_ns": saved_config["configurable"]["checkpoint_ns"]}}).aget_state(
                    {"configurable": {"thread_id": context.request.run_id}})
                values = snapshot.values or values
            messages = [message for message in values.get("messages", [])
                        if message.id not in previous and not message.additional_kwargs.get("lc_source")]
            self._sources(plan, messages, context)
            if name in {"execute", "report"}:
                self._block_unfinished(plan, context, "执行尚未取得可验收的步骤结果：" + "；".join(plan.unresolved))
            return {**{key: values[key] for key in _MEMORY_KEYS if key in values},
                    "messages": messages, **self._save(plan, context), "stage_failure": reason,
                    "references": list(execution.references_snapshot())[len(state.get("references", [])):],
                    "observation_bytes": max(0, execution.observation_bytes - state.get("observation_bytes", 0))}

        try:
            child = self.graphs[name]
            self.active_child = (child, {"configurable": {"thread_id": context.request.run_id,
                                                         "checkpoint_ns": child_config["configurable"]["checkpoint_ns"]}})
            saved = await child.aget_state(child_config) if self.persisted else None
            if saved is not None and saved.values:
                execution.restore(saved.values.get("execution_ledger", {}))
                _budget(context).restore(saved.values.get("model_ledger", {}))
                resumed_config = await aprepare_react_resume(child, child_config, saved)
                # update_state returns a checkpoint pointer, not the enclosing
                # node's runtime configuration. Preserve that invocation's
                # inherited context when resuming the nested agent.
                child_config = {**child_config, "configurable": {
                    **child_config["configurable"], **resumed_config.get("configurable", {})}}
                saved = await child.aget_state(child_config)
                execution.restore(saved.values.get("execution_ledger", {}))
                # A started call without a committed observation is uncertain,
                # not permission to repeat its side effect. Close its original
                # identity with an explicit failure and preserve its consumed
                # allowance. The official agent decides whether a new call is
                # justified; writes must first verify the previous outcome.
                answered = {message.tool_call_id for message in saved.values.get("messages", []) if isinstance(message, ToolMessage)}
                uncertain_ids = execution.used_call_ids - execution.completed_call_ids - answered
                pending_calls = [call for message in saved.values.get("messages", []) if isinstance(message, AIMessage)
                                 for call in message.tool_calls if call["id"] in uncertain_ids]
                if pending_calls:
                    receipts = [ToolMessage(id="interrupted-" + call["id"], tool_call_id=call["id"], name=call["name"], status="error",
                        content="此前调用已开始但运行中断，没有已提交结果。结果未确认，不能宣称完成；写入必须先核验原操作结果，再决定是否发起新的调用。",
                        additional_kwargs={"agent_core_interrupted": True}) for call in pending_calls]
                    before_bytes = execution.observation_bytes
                    try:
                        execution.commit_result(None, observation_size=sum(len(str(message.content).encode("utf-8")) for message in receipts))
                    except AgentLimitReachedError:
                        pass  # Keep the uncertainty receipt even if its accounting exhausted the allowance.
                    for call in pending_calls:
                        execution.complete_attempt(call["id"])
                    resumed_config = await child.aupdate_state(child_config, {"messages": receipts,
                        "execution_ledger": execution.snapshot(), "model_ledger": _budget(context).snapshot(),
                        "observation_bytes": execution.observation_bytes - before_bytes}, as_node="ReActMiddleware.after_model")
                    child_config = {**child_config, "configurable": {
                        **child_config["configurable"], **resumed_config.get("configurable", {})}}
                if context.memory is not None:
                    restore_memory_binding(context.memory, saved.values)
                child_input = None
            else:
                child_input = initial
            async for kind, values in child.astream(child_input, config=child_config, context=context, stream_mode=["values", "custom"],
                                            durability="sync" if self.persisted else None):
                if kind == "custom":
                    # Only root publication may emit answers or task/plan views.
                    # Official middleware also emits private, untyped custom data.
                    if isinstance(values, Mapping) and isinstance(values.get("kind"), str) and values["kind"] in _STAGE_EVENT_KINDS:
                        identity = {"activityScopeId": initial["activity_scope_id"], "activityLabel": initial["activity_label"]}
                        # Older checkpoints lack the optional scope channels.
                        # The parent still knows this invocation's public identity.
                        event = {**values, **identity}
                        if values.get("kind") == "step.state" and isinstance(values.get("payload"), Mapping):
                            event["payload"] = {**values["payload"], **identity}
                        runtime.stream_writer(event)
                else:
                    child_values = dict(values)
            result = child_values
            public_output = (result.get("plan_rubric", {}) if name == "final" else
                             result.get("final_answer", "") if name == "execute" else result.get("structured_response"))
            _validate_public_strings(public_output, self.policy, tuple(execution.sensitive_values))
        except OutputValidationError:
            plan.phase = "partial"
            plan.unresolved = ["最终候选未通过引用或敏感内容检查" if name == "final" else "阶段产物未通过敏感内容检查"]
            return await failure_update("candidate_validation" if name == "final" else "stage_output_validation"), None, []
        except AgentInputBudgetError as error:
            plan.phase = "partial" if any(step.status == "completed" for step in plan.steps) else "failed"
            plan.unresolved = [error.public_message + "已保留计划、已提交证据和累计额度，尚未完成验收。"]
            return await failure_update(error.code), None, []
        except (AgentLimitReachedError, AgentUpstreamError, ValidationError) as error:
            plan.phase = "partial" if any(step.status == "completed" for step in plan.steps) else "failed"
            plan.unresolved = [error.public_message if isinstance(error, StructuredOutputError) else
                               "模型或执行额度耗尽" if isinstance(error, AgentLimitReachedError) else "本阶段未取得有效结果"]
            return await failure_update(type(error).__name__), None, []
        # Child graph checkpoints are official nested checkpoints. Copy only
        # root-owned deltas; copying cumulative additive channels doubles them.
        messages = [message for message in result.get("messages", []) if message.id not in previous]
        messages = [message for message in messages if not message.additional_kwargs.get("lc_source")]
        sources = self._sources(plan, messages, context)
        updates = {key: result[key] for key in _MEMORY_KEYS if key in result}
        updates.update({"messages": messages, "references": list(execution.references_snapshot())[len(state.get("references", [])):],
                        "observation_bytes": max(0, execution.observation_bytes - state.get("observation_bytes", 0)),
                        **self._save(plan, context), "stage_failure": None})
        if name in {"execute", "final"}:
            return updates, result, sources
        output = result.get("structured_response")
        if output is None:
            plan.phase = "partial" if any(step.status == "completed" for step in plan.steps) else "failed"
            plan.unresolved = ["本阶段没有返回符合约定的结构化结果"]
            updates.update({**self._save(plan, context), "stage_failure": "missing_structured_result"})
        return updates, output, sources

    async def explore(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        updates, output, _ = await self._invoke("explore", state, runtime, config)
        plan = parse_plan(updates["plan_state"])
        assert plan is not None
        if output is None:
            return {**updates, "route": "publish"}
        output = ExplorationResult.model_validate(output)
        reason = "根据本轮输入形成问题" if output.kind == "questions" else "根据本轮输入更新可执行计划"
        if plan.revisions:
            self._revision(plan, reason)
        else:
            plan.revision_reason = reason
            plan.revisions.append({"revision": plan.revision, "reason": reason})
        plan.questions, plan.unresolved = output.questions, output.unresolved
        if output.kind == "questions":
            plan.phase = "clarifying"
        else:
            self._apply_draft(plan, output.plan)
            plan.phase = "ready"
        return {**updates, **self._save(plan, runtime.context), "route": "publish"}

    def _apply_draft(self, plan: PlanState, draft: PlanDraft | None) -> None:
        if draft is None:
            raise ValueError("A ready plan needs a complete draft")
        completed = [step for step in plan.steps if step.status == "completed"]
        completed_by_id = {step.id: step for step in completed}
        proposed = []
        for item in draft.steps:
            if item.id in completed_by_id:
                old = completed_by_id[item.id]
                if item.model_dump() != {key: getattr(old, key) for key in type(item).model_fields}:
                    raise ValueError("A completed step cannot be rewritten")
            else:
                proposed.append(PlanStep(**item.model_dump()))
        steps = [*completed, *proposed]
        validate_steps(steps, maximum=self.max_steps)
        for key, value in plan_scope(draft).items():
            setattr(plan, key, deepcopy(value))
        plan.steps = steps
        plan.questions, plan.unresolved = [], []
        plan.active_step_id = None

    @staticmethod
    def _record_step(plan: PlanState, step: PlanStep, context: Any) -> None:
        plan.step_history.append({"stepId": step.id, "status": step.status, "result": step.result,
                                  "unresolved": list(step.unresolved), "sourceIds": list(step.source_ids),
                                  "revision": plan.revision, "runId": context.request.run_id})

    @staticmethod
    def _activity_scope(plan: PlanState, stage: str) -> str:
        # These are public workflow identities, independent of graph task UUIDs
        # and checkpoint namespaces. A resumed invocation retains its scope.
        identity = json.dumps([plan.plan_id, plan.revision, stage, plan.replans], separators=(",", ":"))
        return "pa-" + hashlib.sha256(identity.encode()).hexdigest()[:24]

    def _block_unfinished(self, plan: PlanState, context: Any, reason: str) -> None:
        for step in plan.steps:
            if step.status == "completed":
                continue
            step.status = "blocked"
            step.unresolved = list(dict.fromkeys([*step.unresolved, reason]))
            self._record_step(plan, step, context)

    async def execute(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        plan = self._restore(state, runtime.context)
        runtime.context.cancellation.raise_if_cancelled()
        if plan_scope(plan) != plan.approved_scope:
            raise AgentContextError("Execution plan exceeds its approved scope")
        if all(step.status == "completed" for step in plan.steps):
            return {**self._save(plan, runtime.context), "route": "validate"}
        # Only Todo represents the agent's internal work order. The controller
        # does not infer individual completion from tool calls or Todo labels.
        updates, result, _ = await self._invoke("execute", state, runtime, config)
        if result is None:
            return {**updates, "route": "publish"}
        return {**updates, "execution_answer": result.get("final_answer", ""), "route": "report"}

    async def report(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        updates, output, _ = await self._invoke("report", state, runtime, config)
        plan = parse_plan(updates["plan_state"])
        assert plan is not None
        if output is None:
            return {**updates, "route": "publish"}
        report = ExecutionReport.model_validate(output)
        expected = {step.id for step in plan.steps if step.status != "completed"}
        if {item.id for item in report.steps} != expected:
            plan.phase = "partial" if any(step.status == "completed" for step in plan.steps) else "failed"
            plan.unresolved = ["执行报告必须完整覆盖未完成步骤，不得改写已完成步骤或新增步骤"]
            self._block_unfinished(plan, runtime.context, plan.unresolved[0])
            return {**updates, **self._save(plan, runtime.context), "route": "publish"}
        by_id = {item.id: item for item in report.steps}
        completed = {step.id for step in plan.steps if step.status == "completed"}
        for step in plan.steps:
            if step.id not in by_id:
                continue
            item = by_id[step.id]
            step.result, step.status = item.result, item.status
            step.unresolved = list(item.unresolved)
            unknown = set(item.source_ids) - set(plan.sources)
            step.source_ids = [source for source in item.source_ids if source in plan.sources]
            if unknown:
                step.status = "failed"
                step.unresolved.append("步骤结果引用了不可回查的来源")
            if step.status == "completed" and (step.unresolved or not step.source_ids or not set(step.depends_on) <= completed):
                step.status = "blocked"
                if not step.source_ids:
                    step.unresolved.append("步骤缺少可回查的证据来源")
                if not set(step.depends_on) <= completed:
                    step.unresolved.append("步骤依赖尚未完成")
            if step.status == "completed":
                completed.add(step.id)
            self._record_step(plan, step, runtime.context)
            runtime.stream_writer({"kind": "plan.step", "stepId": step.id,
                                   "activityScopeId": self._activity_scope(plan, "steps"),
                                   "activityLabel": "计划步骤",
                                   "payload": {"planId": plan.plan_id, "revision": plan.revision, "status": step.status}})
        plan.unresolved = list(dict.fromkeys([*(issue for step in plan.steps for issue in step.unresolved), *report.new_conditions]))
        plan.phase = "executing"
        needs_revision = len(completed) != len(plan.steps) or report.needs_replan or report.new_conditions
        return {**updates, **self._save(plan, runtime.context), "route": "replan" if needs_revision else "validate"}

    async def replan(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        plan = self._restore(state, runtime.context)
        if plan.replans >= self.max_replans:
            plan.phase = "partial"
            plan.unresolved = list(dict.fromkeys([*plan.unresolved, "达到剩余计划调整次数上限"]))
            return {**self._save(plan, runtime.context), "route": "publish"}
        plan.replans += 1
        current = {**state, **self._save(plan, runtime.context)}
        updates, output, _ = await self._invoke("replan", current, runtime, config)
        plan = parse_plan(updates["plan_state"])
        assert plan is not None
        if output is None:
            return {**updates, "route": "publish"}
        decision = ReplanDecision.model_validate(output)
        self._revision(plan, decision.reason)
        route = "publish"
        if decision.kind == "continue":
            completed = [step for step in plan.steps if step.status == "completed"]
            if {step.id for step in completed} & {step.id for step in decision.remaining_steps}:
                raise AgentContextError("Automatic replanning cannot replace completed steps")
            proposed = [*completed, *(PlanStep(**step.model_dump()) for step in decision.remaining_steps)]
            validate_steps(proposed, maximum=self.max_steps)
            if decision.proposed_plan is not None or decision.questions:
                raise AgentContextError("Automatic replanning cannot modify confirmed scope")
            plan.steps, plan.unresolved = proposed, decision.unresolved
            plan.phase, plan.active_step_id = "executing", None
            route = "execute"
        elif decision.kind == "questions":
            if not decision.questions:
                raise ValueError("Replanning questions cannot be empty")
            plan.phase, plan.questions, plan.unresolved = "clarifying", decision.questions, decision.unresolved
            plan.approved_revision, plan.approved_scope = None, None
        elif decision.kind == "confirm_scope":
            self._apply_draft(plan, decision.proposed_plan)
            plan.phase = "ready"
            plan.approved_revision, plan.approved_scope = None, None
        else:
            plan.phase = "partial"
            plan.unresolved = decision.unresolved or [decision.reason]
        return {**updates, **self._save(plan, runtime.context), "route": route}

    def validate(self, state: PlanExecuteState, runtime: Runtime) -> dict[str, Any]:
        plan = self._restore(state, runtime.context)
        gaps = list(plan.unresolved)
        if not plan.acceptance:
            gaps.append("计划缺少验收标准")
        for step in plan.steps:
            if step.status != "completed" or not step.result.strip() or step.unresolved:
                gaps.append(f"步骤 {step.id} 尚未完成")
            if not step.source_ids or not set(step.source_ids) <= set(plan.sources):
                gaps.append(f"步骤 {step.id} 缺少可回查证据")
        if gaps:
            plan.phase = "partial"
            plan.unresolved = list(dict.fromkeys(gaps))
            plan.validation = {"status": "deterministic_failed", "criteria": [], "iterations": 0,
                               "unresolved": plan.unresolved, "assessment": "model_assessment"}
        return {**self._save(plan, runtime.context), "route": "replan" if gaps else "final_review"}

    async def final_review(self, state: PlanExecuteState, runtime: Runtime, config: RunnableConfig) -> dict[str, Any]:
        updates, result, _ = await self._invoke("final", state, runtime, config)
        plan = parse_plan(updates["plan_state"])
        assert plan is not None
        if result is None:
            deterministic = updates.get("stage_failure") == "candidate_validation"
            plan.validation = {"status": "deterministic_failed" if deterministic else "grader_error", "criteria": [], "iterations": 0,
                               "unresolved": plan.unresolved, "assessment": "model_assessment"}
            return {**updates, **self._save(plan, runtime.context), "route": "replan" if deterministic else "publish"}
        rubric = result.get("plan_rubric", {})
        evaluations = rubric.get("evaluations", [])
        latest = evaluations[-1] if evaluations else {}
        status = rubric.get("status") or "grader_error"
        criteria = latest.get("criteria", [])
        expected_names = {f"A{index}": f"A{index}: {criterion}" for index, criterion in enumerate(plan.acceptance, 1)}
        expected = set(expected_names)
        actual = [str(item.get("name", "")).split(":", 1)[0].strip() for item in criteria]
        covered = (len(actual) == len(expected) and set(actual) == expected
                   and all(item.get("passed") is True and str(item.get("name", "")).strip() == expected_names.get(identity)
                           for item, identity in zip(criteria, actual)))
        if status == "satisfied" and not covered:
            status = "needs_revision"
        unresolved = [str(item.get("gap") or item.get("name")) for item in criteria if item.get("passed") is not True]
        if not covered and status == "needs_revision":
            unresolved.append("模型复核未逐项覆盖冻结验收标准")
        frozen = {f"A{index}": criterion for index, criterion in enumerate(plan.acceptance, 1)}
        plan.validation = {"status": status,
                           "criteria": [{"id": str(item.get("name", "")).split(":", 1)[0].strip(),
                                         "criterion": frozen.get(str(item.get("name", "")).split(":", 1)[0].strip(), "未知验收标准"),
                                         "passed": item.get("passed") is True,
                                         "gap": str(item.get("gap", ""))} for item in criteria],
                           "iterations": rubric.get("iterations", len(evaluations)),
                           "unresolved": unresolved, "assessment": "model_assessment"}
        plan.phase = "completed" if status == "satisfied" and covered else "partial"
        plan.unresolved = unresolved if plan.phase == "completed" else unresolved or ["最终模型复核未通过"]
        candidate = result.get("final_answer", "")
        return {**updates, **self._save(plan, runtime.context), "final_answer": candidate, "route": "publish"}

    def publish(self, state: PlanExecuteState, runtime: Runtime) -> dict[str, Any]:
        plan = self._restore(state, runtime.context)
        if plan.questions or plan.phase == "ready":
            answer = render_plan_control_text(plan)
        elif plan.phase == "completed":
            answer = state.get("final_answer", "")
        else:
            answer = "任务尚未完成。" + ("\n" + "\n".join(plan.unresolved) if plan.unresolved else "")
            completed = [step for step in plan.steps if step.status == "completed"]
            if completed:
                answer += "\n已完成：\n" + "\n".join(f"{step.goal}：{step.result}" for step in completed)
        policy = self.policy.control_output_policy(state) if plan.phase != "completed" else self.policy
        published = publish_answer({**state, "activity_scope_id": self._activity_scope(plan, "publish"),
                                    "activity_label": "汇总回复"}, runtime, policy, [answer])
        snapshot = self._save(plan, runtime.context)
        runtime.stream_writer({"kind": "plan.updated", "payload": {"plan": project_plan(plan)}})
        return {**snapshot, "final_answer": published, "stop_reason": "plan_" + plan.phase}

    async def finalize_error(self, graph: Any, config: dict[str, Any], context: Any, error: BaseException) -> None:
        """Commit meters at a terminal boundary after the active stream stops.

        This writes only root-owned state through the official Saver API. The
        stable stage namespace preserves the separately checkpointed ReAct work
        if the same Run is explicitly resumed later.
        """
        saved = await graph.aget_state(config)
        if not saved.values.get("plan_state"):
            return
        plan = self._restore(saved.values, context)
        updates = {}
        if self.active_child is not None:
            child, child_config = self.active_child
            child_snapshot = await child.with_config({"configurable": {"checkpoint_ns": child_config["configurable"]["checkpoint_ns"]}}).aget_state(
                {"configurable": {"thread_id": context.request.run_id}})
            values = child_snapshot.values
            updates.update({key: values[key] for key in _MEMORY_KEYS if key in values})
            old_ids = {message.id for message in saved.values.get("messages", [])}
            messages = [message for message in values.get("messages", [])
                        if message.id not in old_ids and not message.additional_kwargs.get("lc_source")]
            self._sources(plan, messages, context)
            updates["messages"] = messages
        if context.memory is not None and context.memory.snapshot is not None:
            updates["memory_release_snapshot"] = memory_binding_checkpoint(context.memory)
        execution = context.tool_runtime.execution
        updates["references"] = list(execution.references_snapshot())[len(saved.values.get("references", [])):]
        updates["observation_bytes"] = max(0, execution.observation_bytes - saved.values.get("observation_bytes", 0))
        plan.phase = "cancelled" if isinstance(error, AgentCancelled) else (
            "partial" if any(step.status == "completed" for step in plan.steps) else "failed")
        reason = "本次运行已取消" if plan.phase == "cancelled" else "本次运行中断，尚未完成验收"
        plan.unresolved = list(dict.fromkeys([*plan.unresolved, reason]))
        self._block_unfinished(plan, context, reason)
        await graph.aupdate_state(config, {**updates, **self._save(plan, context), "stop_reason": "plan_" + plan.phase})


def compile_plan_execute_graph(
    model: Any, tools: Sequence[BaseTool], *, stream_chat_message: Callable[..., Any], policy: ReActPolicy,
    context_schema: type[Any] = ReActContext, action: PlanInput | None = None,
    exploration_tools: Sequence[BaseTool] | None = None, middleware: Sequence[AgentMiddleware] = (),
    memory: Any = None, checkpointer: Any = None, store: Any = None,
    max_steps: int = 8, max_replans: int = 3, max_rubric_iterations: int = 3, max_model_calls: int | None = 64,
    context_summarizer: Any = None, prepare_context: Callable[..., Any] | None = None,
    native_authorizer: Any = None,
) -> Any:
    """Compile a planning extension around whole-task ReAct and final review.

    ``context.model_budget`` must be a ``PlanModelBudget`` shared by stages.
    Exploration has no domain tools unless the host explicitly supplies its
    read-only subset. Child agents inherit the root's official Saver namespace.
    ``max_model_calls=None`` disables this workflow's model-call cap. A finite
    cap explicitly supplied by the shared context budget still applies.
    """
    if (min(max_steps, max_replans, max_rubric_iterations) < 1 or max_steps > 64
            or (max_model_calls is not None and max_model_calls < 1)):
        raise ValueError("Plan limits must be positive and max_steps cannot exceed 64")
    workflow = _PlanWorkflow(model, tools, exploration_tools=exploration_tools or (),
        stream_chat_message=stream_chat_message, policy=policy, context_schema=context_schema,
        middleware=middleware, memory=memory, store=store, action=action,
        max_steps=max_steps, max_replans=max_replans, max_rubric_iterations=max_rubric_iterations,
        max_model_calls=max_model_calls, checkpointer=checkpointer,
        context_summarizer=context_summarizer, prepare_context=prepare_context, native_authorizer=native_authorizer)
    graph = StateGraph(PlanExecuteState, context_schema=context_schema)
    for name in ("prepare", "prepare_context", "explore", "execute", "report", "replan", "validate", "final_review", "publish"):
        graph.add_node(name, getattr(workflow, name))
    graph.add_edge(START, "prepare")
    for name in ("prepare", "prepare_context", "explore", "execute", "report", "replan", "validate", "final_review"):
        graph.add_conditional_edges(name, lambda state: state["route"],
            {target: target for target in ("prepare_context", "explore", "execute", "report", "replan", "validate", "final_review", "publish")})
    graph.add_edge("publish", END)
    compiled = graph.compile(checkpointer=checkpointer, store=store, name="plan_and_execute")
    compiled._finalize_plan_error = workflow.finalize_error
    return compiled


async def afinalize_plan_error(graph: Any, config: dict[str, Any], context: Any, error: BaseException) -> None:
    """WorkflowExecution error hook for persisted plan graphs."""
    await graph._finalize_plan_error(graph, config, context, error)


__all__ = ["compile_plan_execute_graph", "finalize_plan_error"]


def finalize_plan_error(graph, config, context, error):
    return run_sync(afinalize_plan_error(graph, config, context, error))

finalize_plan_error.__async__ = afinalize_plan_error
