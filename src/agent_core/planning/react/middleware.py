"""Application contracts around LangChain's standard model/tool loop.

No routing graph is maintained here. Hooks enforce per-Run budgets and safe
events; LangChain determines when to execute tools or finish the agent.
"""

from collections.abc import Callable
import json
import re
from typing import Any
from agent_core.runtime.streaming import call_async, call_port, run_sync

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse, hook_config
from langchain.agents.structured_output import ToolStrategy
from langchain.agents.middleware.types import ExtendedModelResponse
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from agent_core.contracts.errors import AgentLimitReachedError, AgentUpstreamError, StructuredOutputError, ToolCallingUnsupportedError
from agent_core.context.budget import archive_messages, history_source_prompt, select_context
from agent_core.context.summary import ContextSummarizer, prepare_context_view
from agent_core.models.langchain import StreamingChatModelAdapter
from agent_core.planning.react.serial import SerialToolCalls
from agent_core.planning.react.output import emit_step, publish_answer
from agent_core.contracts.events import activity_label, activity_scope
from agent_core.planning.react.policy import ReActPolicy
from agent_core.planning.react.state import ReActState
from agent_core.tools.native import NativeToolGovernance


_EXECUTION_LIMITS = {"tool_call_limit", "observation_limit", "reference_limit"}
_MAX_STRUCTURED_REPAIRS = 2


def _structured_tools(response_format: Any) -> list[dict[str, Any]]:
    """Include the official response schema in the final model input estimate.

    These functions are model output parsers, not service tools. LangChain adds
    them after middleware, so they must be counted without dispatching or
    charging them as business I/O. Provider-native schemas use the same bound.
    """
    if response_format is None:
        return []
    strategy = response_format if isinstance(response_format, ToolStrategy) else ToolStrategy(
        getattr(response_format, "schema", response_format)
    )
    return [{"type": "function", "function": {"name": spec.name,
             "description": spec.description, "parameters": spec.json_schema}}
            for spec in strategy.schema_specs]


def _stop_reason(state: ReActState, context: Any) -> str | None:
    execution = context.tool_runtime.execution
    active_limits = _EXECUTION_LIMITS if execution.max_observation_bytes is not None else _EXECUTION_LIMITS - {"observation_limit"}
    if execution.stop_reason in active_limits:
        return execution.stop_reason
    if state.get("stop_reason") in active_limits:
        return state["stop_reason"]
    if state.get("planning_rounds", 0) >= context.settings.max_planning_rounds:
        return "planning_limit"
    if state.get("tool_call_count", 0) >= min(context.settings.max_tool_calls, execution.max_tool_calls):
        return "tool_call_limit"
    if (execution.max_observation_bytes is not None and
            max(state.get("observation_bytes", 0), execution.observation_bytes) >= execution.max_observation_bytes):
        return "observation_limit"
    return None


class ReActMiddleware(AgentMiddleware):
    """One instance per Run; only the current provider's text chunks are buffered."""

    state_schema = ReActState

    def __init__(self, policy: ReActPolicy, model_calls: "ReActModelMiddleware", *, components: Any = None,
                 publish_final: bool = True, access: Any = None, native_authorizer: Any = None) -> None:
        self.policy = policy
        self.model_calls = model_calls
        self.publish_final = publish_final
        self._tools = SerialToolCalls()
        self.access = access
        async def authorize_memory(name, args):
            if components and name in components.tool_metadata:
                return await call_async(components.authorize_tool, name, args)
            return None
        native_authorize = authorize_memory if components else None
        self._native = NativeToolGovernance(metadata=access.metadata if access else (components.tool_metadata if components else None),
                                            authorize=native_authorize, access=access, authorizer=native_authorizer)
        self.model_calls.tool_metadata = self._native.metadata

    def before_agent(self, state: ReActState, runtime: Any) -> None:
        runtime.context.cancellation.raise_if_cancelled()
        emit_step(runtime, "context_validation", self.policy.text("context_step"), "running", state=state)
        self.policy.validate_context(runtime.context)
        self._restore(state, runtime.context)
        if self.access is not None:
            self.access.restore(state, runtime.context)
        emit_step(runtime, "context_validation", self.policy.text("context_step"), "succeeded", state=state)

    @staticmethod
    def _restore(state: Any, context: Any) -> None:
        execution = context.tool_runtime.execution
        execution.max_external_tool_calls = context.settings.max_external_tool_calls
        execution.max_memory_tool_calls = context.settings.max_memory_tool_calls
        execution.max_discovery_tool_calls = context.settings.max_discovery_calls
        execution.max_retry_attempts = context.settings.max_tool_retry_slots
        execution.max_extra_retries = context.settings.max_tool_extra_retries
        execution.retries_enabled = context.settings.tool_retry_enabled
        execution.discovery_enabled = context.settings.tool_discovery_enabled
        execution.restore(state.get("execution_ledger", {}))
        model_budget = getattr(context, "model_budget", None)
        if model_budget is not None:
            model_budget.restore(state.get("model_ledger", {}))

    def before_model(self, state: ReActState, runtime: Any) -> None:
        self._restore(state, runtime.context)

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: ReActState, runtime: Any) -> dict[str, Any]:
        context = runtime.context
        context.cancellation.raise_if_cancelled()
        response = next(message for message in reversed(state["messages"]) if isinstance(message, AIMessage))
        answered = {message.tool_call_id for message in state["messages"] if isinstance(message, ToolMessage)}
        reason = _stop_reason(state, context)
        number = state.get("planning_rounds", 0) + 1
        schema_names = self.model_calls.structured_tool_names
        business_calls = [call for call in response.tool_calls if call["name"] not in schema_names]
        invalid_schema = bool(response.invalid_tool_calls) and all(
            not call.get("name") or call["name"] in schema_names for call in response.invalid_tool_calls)
        missing_schema = not response.tool_calls and not response.invalid_tool_calls
        schema_attempt = not business_calls and bool(missing_schema or response.tool_calls or invalid_schema)
        if schema_names and state.get("structured_response") is None and schema_attempt:
            retries = state.get("structured_output_retries", 0)
            if reason == "planning_limit" or retries >= _MAX_STRUCTURED_REPAIRS:
                raise StructuredOutputError(
                    truncated=response.response_metadata.get("finish_reason") == "length",
                    budget_exhausted=reason == "planning_limit",
                    limit_message=context.tool_runtime.execution.limit_message,
                )
            emit_step(runtime, f"agent_decision_{number}", f"自主规划 · 第 {number} 轮", "succeeded", state=state,
                      summary="未取得有效结构化结果，将在剩余额度内修复")
            updates = {"planning_rounds": number, "structured_output_retries": retries + 1,
                       "stop_reason": None, "final_answer": "", "jump_to": "model",
                       "execution_ledger": context.tool_runtime.execution.snapshot(), **_model_ledger(context)}
            if invalid_schema:
                # Truncated JSON cannot be replayed as an unanswered provider
                # tool call. The retry directive lives in the system prompt;
                # do not manufacture a HumanMessage or a business tool receipt.
                updates["messages"] = [AIMessage(id=response.id, content=str(response.text),
                    usage_metadata=response.usage_metadata, response_metadata=response.response_metadata)]
            return updates
        if reason:
            business_calls = [call for call in response.tool_calls
                              if call["name"] not in self.model_calls.structured_tool_names]
            if business_calls or response.invalid_tool_calls:
                raise ToolCallingUnsupportedError("最终回答意外返回工具调用")
            return {"stop_reason": reason, "final_answer": str(response.text),
                    **({"structured_output_retries": 0} if schema_names and state.get("structured_response") is not None else {})}
        calls = [call for call in response.tool_calls
                 if call["name"] not in self.model_calls.structured_tool_names]
        if response.tool_calls and not calls and not response.invalid_tool_calls:
            # The official response-format handler has already validated the
            # schema or appended a repair ToolMessage. Its standard routing
            # either completes or retries; never send these calls to ToolNode.
            ready = state.get("structured_response") is not None
            emit_step(runtime, f"agent_decision_{number}", f"自主规划 · 第 {number} 轮", "succeeded", state=state,
                      summary="已形成结构化结果" if ready else "修正结构化结果")
            return {"planning_rounds": number, "stop_reason": "model_ready" if ready else None,
                    "execution_ledger": context.tool_runtime.execution.snapshot(),
                    **({"structured_output_retries": 0} if ready else {})}
        if response.invalid_tool_calls:
            reserved = {str(call.get("id") or "") for call in calls} | set(state.get("used_tool_call_ids", []))
            for index, _invalid in enumerate(response.invalid_tool_calls, 1):
                call_id = f"invalid-{number}-{index}"
                while call_id in reserved:
                    call_id += "-retry"
                reserved.add(call_id)
                calls.append({"name": "__invalid_tool_call__", "args": {}, "id": call_id, "type": "tool_call"})
            # Rebuild rather than copy: some providers also keep raw calls in
            # additional_kwargs or content blocks, including malformed args.
            response = AIMessage(id=response.id, content="", tool_calls=calls,
                                 usage_metadata=response.usage_metadata)
        step_id = f"agent_decision_{number}"
        if not calls:
            emit_step(runtime, step_id, f"自主规划 · 第 {number} 轮", "succeeded", state=state, summary="模型完成回答")
            return {"planning_rounds": number, "stop_reason": "model_ready", "final_answer": str(response.text)}
        count = state.get("tool_call_count", 0)
        if count + len(calls) > min(context.settings.max_tool_calls, context.tool_runtime.execution.max_tool_calls):
            emit_step(runtime, step_id, f"自主规划 · 第 {number} 轮", "succeeded", state=state,
                      summary="计划超过剩余工具调用上限，基于现有证据作答")
            # Replace by ID so rejected calls never reach tools or leave an
            # unmatched tool-call message in the provider history.
            response = AIMessage(id=response.id, content="本批工具调用超过剩余上限，未执行。",
                                 usage_metadata=response.usage_metadata)
            return {"messages": [response], "planning_rounds": number,
                    "stop_reason": "tool_call_limit", "jump_to": "model"}
        used_ids = list(state.get("used_tool_call_ids", []))
        seen_ids = set(used_ids)
        for call in calls:
            call_id = str(call.get("id") or "")
            if not call_id:
                raise ToolCallingUnsupportedError(self.policy.text("missing_tool_call_id"))
            if context.memory is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", call_id):
                raise ToolCallingUnsupportedError("工具调用标识不能安全用于官方结果文件路径")
            if call_id in seen_ids:
                raise ToolCallingUnsupportedError(self.policy.text("duplicate_tool_call_id"))
            seen_ids.add(call_id)
            used_ids.append(call_id)
        self._tools.begin([call["id"] for call in calls])
        self._tools.prepare([call["id"] for call in calls], answered)
        # Official Todo validation may answer calls before ToolNode dispatch.
        # These rejected attempts still consume the same recoverable budget.
        execution = context.tool_runtime.execution
        early_observation_bytes = 0
        for call in calls:
            if call["id"] in answered and call["id"] not in execution.used_call_ids:
                try:
                    execution.begin_attempt(call_id=call["id"], tool_name=call["name"],
                                            max_calls_for_tool=None, category="memory")
                    size = sum(len(str(message.content).encode("utf-8")) for message in state["messages"]
                               if isinstance(message, ToolMessage) and message.tool_call_id == call["id"])
                    execution.commit_result(None, observation_size=size)
                    early_observation_bytes += size
                    execution.complete_attempt(call["id"])
                except AgentLimitReachedError:
                    execution.complete_attempt(call["id"])
                    if execution.stop_reason in _EXECUTION_LIMITS:
                        break
        # This node commits before ToolNode I/O (durability=sync). Pending
        # retry calls are never allowed a fresh allowance after process loss.
        if execution.retries_enabled and self.access is not None:
            for call in calls:
                if call["id"] in answered or call["name"] not in self.model_calls.tool_names:
                    continue
                definition = self.access.retry_definition(call["name"])
                metadata = self.access.metadata.get(call["name"], {})
                retry_protocol = definition is not None and definition.retry_policy.mode == "safe_read"
                try:
                    execution.reserve_call(call_id=call["id"], tool_name=call["name"],
                        max_calls_for_tool=definition.max_calls_per_run if definition else None,
                        budget_group=definition.budget_group if definition else metadata.get("budget_group", "external"),
                        max_extra_retries=definition.retry_policy.max_extra_retries if retry_protocol else 0,
                        arguments=call.get("args", {}), tool_version=definition.version if definition else metadata.get("version"),
                        retry_protocol=retry_protocol)
                except AgentLimitReachedError:
                    # Tool governance will return the ordinary bounded denial.
                    continue
        names = []
        for index, call in enumerate(calls, count + 1):
            name = call["name"] if call["name"] in self.model_calls.tool_names else "未注册工具"
            names.append(name)
            runtime.stream_writer({"kind": "tool.planned", "callId": call["id"],
                                   "stepId": f"tool_{index}", "toolName": name,
                                   "activityScopeId": activity_scope(state), "activityLabel": activity_label(state)})
        emit_step(runtime, step_id, f"自主规划 · 第 {number} 轮", "succeeded", state=state, summary="按需选择：" + "、".join(names))
        return {"messages": [response], "planning_rounds": number, "tool_call_count": count + len(calls),
                "used_tool_call_ids": used_ids, "stop_reason": None, "execution_ledger": execution.snapshot(),
                "observation_bytes": early_observation_bytes}

    async def awrap_tool_call(self, request: Any, handler: Callable[..., Any]) -> Any:
        self._restore(request.state, request.runtime.context)
        messages = request.state["messages"]
        response = next(message for message in reversed(messages) if isinstance(message, AIMessage))
        self._tools.prepare([call["id"] for call in response.tool_calls],
                            [*[message.tool_call_id for message in messages if isinstance(message, ToolMessage)],
                             *request.runtime.context.tool_runtime.execution.completed_call_ids])
        return await self._tools.aexecute(
            request.tool_call["id"], request.runtime.context.cancellation,
            lambda: self._native.aexecute(request, lambda call: self._execute_tool(call, handler)),
        )

    async def _execute_tool(self, request: Any, handler: Callable[..., Any]) -> Any:
        if request.tool is not None:
            result = await call_async(handler, request)
            if isinstance(result, Command) and isinstance(result.update, dict):
                # Filesystem middleware may wrap its receipt with file state.
                # Keep the official updates and normalize only known read errors.
                if request.tool.name == "read_file" and "read_file" in self._native.metadata:
                    return Command(update={**result.update, "messages": [
                        self._file_read_error(message, request) or message
                        for message in result.update.get("messages", [])]},
                        goto=result.goto, graph=result.graph, resume=result.resume)
                return result
            if not isinstance(result, ToolMessage) or result.status != "error":
                return result
            if request.tool.name == "read_file" and "read_file" in self._native.metadata:
                normalized = self._file_read_error(result, request)
                if normalized is not None:
                    return normalized
        return ToolMessage(
            content=json.dumps({"ok": False, "errorCode": "invalid_tool_call",
                                "message": self.policy.text("invalid_tool_call")},
                               ensure_ascii=False, separators=(",", ":")),
            tool_call_id=request.tool_call["id"], status="error",
            name="unregistered_tool" if request.tool is None else request.tool.name,
        )

    @staticmethod
    def _file_read_error(message: Any, request: Any) -> ToolMessage | None:
        if not isinstance(message, ToolMessage) or message.status != "error":
            return None
        # Match upstream's exact forms; arbitrary paths/backend text stay private.
        text = str(message.text)
        offset = re.fullmatch(r"Error: Line offset \d+ exceeds file length \((\d+) lines\)", text)
        path = request.tool_call.get("args", {}).get("file_path")
        if offset:
            payload = {"ok": False, "errorCode": "file_read_offset_out_of_range",
                "message": "读取起始行超过文件长度。offset从0开始；limit是源文件行数，长行的续行标记不另算行。单行长结果请通过History按字符分页读取。",
                "totalLines": int(offset[1])}
        elif isinstance(path, str) and text == f"Error: File '{path}' not found":
            payload = {"ok": False, "errorCode": "file_not_found",
                "message": "指定文件不存在。/scratch 临时文件不跨Run保留；请从来源目录使用原始业务观察的 sourceId 回读History，不要猜测路径或反复读取检索回执。"}
        else:
            return None
        return message.model_copy(update={"content": json.dumps(payload, ensure_ascii=False)})

    async def aafter_agent(self, state: ReActState, runtime: Any) -> dict[str, Any]:
        if not self.publish_final:
            # A workflow stage yields data to its parent. Only the parent may
            # publish the final task answer after all validation has completed.
            await call_async(archive_messages, runtime.context.memory, state["messages"])
            message = next((item for item in reversed(state["messages"]) if isinstance(item, AIMessage)), None)
            return {"final_answer": str(message.text) if message else "",
                    "execution_ledger": runtime.context.tool_runtime.execution.snapshot(),
                    **_model_ledger(runtime.context)}
        message = state["messages"][-1]
        if not isinstance(message, AIMessage):
            raise ToolCallingUnsupportedError("最终回答缺少模型消息")
        if message.tool_calls or message.invalid_tool_calls:
            raise ToolCallingUnsupportedError("最终回答仍有未执行工具调用")
        text = str(message.text)
        # A community middleware may revise the response. Preserve original
        # chunk boundaries only while they still describe the final candidate.
        chunks = self.model_calls.answer_chunks
        if "".join(chunks) != text:
            chunks = [text]
        answer = publish_answer(state, runtime, self.policy, chunks)
        await call_async(archive_messages, runtime.context.memory, state["messages"])
        return {"final_answer": answer, "execution_ledger": runtime.context.tool_runtime.execution.snapshot(),
                **_model_ledger(runtime.context)}

    def wrap_tool_call(self, request, handler):
        return run_sync(self.awrap_tool_call(request, handler))

    def after_agent(self, state, runtime):
        return run_sync(self.aafter_agent(state, runtime))

    async def abefore_agent(self, state, runtime):
        return self.before_agent(state, runtime)

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)


def _model_ledger(context: Any) -> dict[str, Any]:
    budget = getattr(context, "model_budget", None)
    return {"model_ledger": budget.snapshot()} if budget is not None else {}


class ReActModelMiddleware(AgentMiddleware):
    """Innermost call adapter, after community model/prompt/retry middleware."""

    state_schema = ReActState

    def __init__(self, policy: ReActPolicy, stream_chat_message: Callable[..., Any], *,
                 archive_history: bool = True, summarizer: Any = None, access: Any = None) -> None:
        self.policy = policy
        self.stream_chat_message = stream_chat_message
        self.archive_history = archive_history
        self.summarizer = summarizer
        self.answer_chunks: list[str] = []
        self.tool_names: set[str] = set()
        self.access = access
        self.structured_tool_names: set[str] = set()
        self.tool_metadata: dict[str, Any] = {}

    def before_model(self, state: Any, runtime: Any) -> None:
        budget = getattr(runtime.context, "model_budget", None)
        if budget is not None:
            budget.restore(state.get("model_ledger", {}))

    def after_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        return _model_ledger(runtime.context)

    async def awrap_model_call(self, request: ModelRequest, handler: Callable[..., ModelResponse]) -> ModelResponse | ExtendedModelResponse:
        context = request.runtime.context
        context.cancellation.raise_if_cancelled()
        reason = _stop_reason(request.state, context)
        number = request.state.get("planning_rounds", 0) + 1
        step_id = "final_answer" if reason else f"agent_decision_{number}"
        emit_step(request.runtime, f"agent_decision_{number}", f"自主规划 · 第 {number} 轮",
                  "skipped" if reason else "running",
                  state=request.state,
                  **({"summary": "已达到执行上限，基于现有证据作答"} if reason else {}))
        chunks: list[str] = []
        length = 0

        def collect_text(text: str) -> None:
            nonlocal length
            length += len(text)
            if length > context.settings.max_answer_characters:
                raise AgentUpstreamError(self.policy.text("answer_too_long"))
            chunks.append(text)

        async def collect(model: Any, messages: Any, **kwargs: Any) -> Any:
            budget = getattr(context, "model_budget", None)
            if budget is not None:
                budget.begin_call()
            result = None
            try:
                result = await call_async(self.stream_chat_message,
                    model, messages, context.cancellation,
                    async_stream_runner=context.async_stream_runner, on_text=collect_text,
                    phase="answer" if reason else "decision", step_id=step_id, **kwargs,
                )
                return result
            finally:
                if budget is not None:
                    budget.record_usage(result)

        delegate = request.model.delegate if isinstance(request.model, StreamingChatModelAdapter) else request.model
        adapter = StreamingChatModelAdapter(delegate=delegate, collect=collect, profile=request.model.profile)
        execution = context.tool_runtime.execution
        for tool in request.tools:
            limit = (getattr(tool, "extras", None) or {}).get("agent_core_max_calls")
            if isinstance(limit, int) and limit > 0:
                execution.tool_limits[tool.name] = limit
        access_update: dict[str, Any] = {}
        directory_prompt = ""
        if self.access is not None:
            tools, directory_prompt, access_update = await self.access.aselect(request, finishing=bool(reason))
        else:
            tools = [] if reason else request.tools
        names = [item.name if hasattr(item, "name") else str(item.get("name", "")) for item in tools]
        self.tool_names = set(names)
        structured_tools = _structured_tools(request.response_format)
        self.structured_tool_names = {tool["function"]["name"] for tool in structured_tools}
        if self.tool_names & self.structured_tool_names:
            raise ToolCallingUnsupportedError("结构化输出名称不能覆盖已注册工具")
        prompt_state = {**request.state, "stop_reason": reason}
        if structured_tools:
            hook = getattr(self.policy, "structured_prompt", None)
            prompt = hook(context, names, sorted(self.structured_tool_names)) if callable(hook) else ""
            prompt += ("\n本次调用必须形成结构化阶段结果。可以先使用本轮允许的工具取得资料；"
                       "完成阶段时必须调用以下输出工具之一：" + "、".join(sorted(self.structured_tool_names)) + "。"
                       "输出工具只提交阶段数据，不执行领域操作，也不代表批准执行任务。"
                       "普通文本、JSON代码块或思考内容均不能代替输出工具调用；请给出简洁完整的参数。")
            if request.state.get("structured_output_retries", 0):
                prompt += "\n前次未取得有效的结构化结果。请缩短内容，按输出工具的schema提交完整结果，不要重复普通文本回答。"
        else:
            prompt = self.policy.final_prompt(prompt_state)
            if not reason:
                prompt = self.policy.planner_prompt(context, names) + "\n\n" + prompt
        if request.system_message is not None:
            prompt = str(request.system_message.text) + "\n\n" + prompt
        if directory_prompt:
            prompt += "\n\n" + directory_prompt
        model_budget = getattr(context, "model_budget", None)
        if model_budget is not None:
            model_public = getattr(model_budget, "public", None)
            prompt += "\n当前实际累计额度（不能通过新步骤或修订重置）：" + json.dumps(
                {**(model_public() if callable(model_public) else {}), **execution.public_budget()}, ensure_ascii=False)
            prompt += ("\n规划前核对完成所需调用能否落在剩余额度内；不能承诺工具不支持或预算不足的完整覆盖。"
                       "maxModelCalls或maxObservationBytes为null表示该资源未设应用层上限，用量仍累计；工具调用上限仍有效。"
                       "若工具额度已用尽，只依据已有来源形成阶段产物并说明缺项；结果输出工具仍可使用。")
        # Grader-generated user-role envelopes are internal evaluation input,
        # never a new statement by the human who owns the Turn.
        sources = await call_async(archive_messages, context.memory, request.messages) if self.archive_history else []
        prompt += history_source_prompt(sources)
        tool_sources = ({message.id: message.additional_kwargs.get("history_source_id") or
                         f"{context.memory.scope.run_id}:{message.id}"
                         for message in request.messages if isinstance(message, ToolMessage) and message.id}
                        if sources and "history_read" in names else {})
        async def read_archived_source(source_id):
            entries = await call_port(context.memory.history, "read", context.memory.scope, [source_id],
                max_characters=context.memory.limits.max_history_characters, offset_characters=0)
            return next((entry for entry in entries if entry.source_id == source_id), None)
        configured_output = (request.model_settings.get("max_completion_tokens") or request.model_settings.get("max_tokens")
                             or getattr(delegate, "max_tokens", None) or getattr(delegate, "max_completion_tokens", None))
        model_settings = dict(request.model_settings)
        if configured_output is None and context.settings.max_input_tokens is not None:
            model_settings["max_tokens"] = context.settings.output_token_reserve
        if self.summarizer is None:
            self.summarizer = ContextSummarizer(delegate, self.stream_chat_message, context)
        prepared = await call_async(prepare_context_view, request.messages, system_message=SystemMessage(prompt), tools=[*tools, *structured_tools],
            state=request.state, context=context, summarizer=self.summarizer,
            archived=bool(sources), output_tokens=configured_output)
        warnings = list(dict.fromkeys(filter(None, [request.state.get("context_warning"), prepared.warning])))
        if warnings:
            prompt += "\n\n上下文覆盖说明：" + "\n".join(warnings)
            emit_step(request.runtime, "context_summary", "整理对话上下文", "failed", state=request.state, summary=warnings[-1])
        elif prepared.changed:
            emit_step(request.runtime, "context_summary", "整理对话上下文", "succeeded", state=request.state,
                      summary="已生成带来源的派生摘要；原始消息保留")
        selection = await call_async(select_context, prepared.messages, system_message=SystemMessage(prompt), tools=[*tools, *structured_tools],
                                   settings=context.settings, archived=bool(sources), output_tokens=configured_output,
                                   tool_sources=tool_sources, history_reader=read_archived_source if tool_sources else None)
        request.runtime.stream_writer({"kind": "context.selected", "inputTokens": selection.input_tokens,
            "counting": selection.counting, "removedMessages": selection.removed_messages,
            "compactedMessages": selection.compacted_messages,
            "inputLimit": selection.input_limit, "outputReserve": configured_output or context.settings.output_token_reserve})
        choice = None if reason else request.tool_choice
        forced_name = choice if isinstance(choice, str) and choice not in {"auto", "any", "required", "none"} else None
        if isinstance(choice, dict):
            forced_name = choice.get("function", {}).get("name") or choice.get("name")
        if forced_name is not None and forced_name not in self.tool_names | self.structured_tool_names:
            raise ToolCallingUnsupportedError("Forced tool choice is not in the visible tool set")
        context.tool_runtime.execution.begin_model_request()
        response = await call_async(handler, request.override(
            model=adapter, tools=tools, tool_choice=choice,
            system_message=SystemMessage(prompt),
            messages=selection.messages,
            model_settings=model_settings,
        ))
        # Mark official output-parser receipts before the model node emits its
        # update, so workflow consumers never display them as service queries.
        if self.structured_tool_names:
            response = ModelResponse(result=[message.model_copy(update={"additional_kwargs": {
                **message.additional_kwargs, "agent_output_schema": True,
            }}) if isinstance(message, ToolMessage) and message.name in self.structured_tool_names
                else message for message in response.result],
                structured_response=response.structured_response)
        message = response.result[-1]
        self.answer_chunks = chunks if chunks else [str(message.text)]
        updates = {**access_update, "execution_ledger": context.tool_runtime.execution.snapshot()}
        if prepared.changed or self.summarizer.calls != request.state.get("summary_calls", 0):
            updates.update(context_summary=prepared.summary or {}, summary_calls=self.summarizer.calls)
        if updates:
            return ExtendedModelResponse(model_response=response, command=Command(update=updates))
        return response

    def wrap_model_call(self, request, handler):
        return run_sync(self.awrap_model_call(request, handler))

    async def abefore_model(self, state, runtime):
        return self.before_model(state, runtime)

    async def aafter_model(self, state, runtime):
        return self.after_model(state, runtime)
