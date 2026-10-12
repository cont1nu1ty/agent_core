"""Execute public core APIs with scripted models and no application imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace
from typing import TypedDict
import unittest

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from agent_core.config import AgentConfig
from agent_core.context.builder import build_messages
from agent_core.contracts.context import RunContext
from agent_core.contracts.errors import OutputValidationError, ToolCallingUnsupportedError
from agent_core.contracts.runs import RunRequest, RunResult
from agent_core.memory.history import HistoryMessage
from agent_core.memory.stores.snapshot import SnapshotHistory
from agent_core.observability.observers.in_memory import InMemoryUsageObserver
from agent_core.observability.observers.noop import NoopUsageObserver
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.runtime.budget import ExecutionBudget
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.service import AgentService
from agent_core.tools import ToolExecutor, ToolRegistry, ToolResult, ToolSpec
from agent_core.tools.contracts import AuthorizationDecision
from agent_core.tools.langchain import build_langchain_tools
from agent_core.workflows.contracts import (
    RunResources,
    WorkflowDefinition,
    WorkflowExecution,
)
from agent_core.workflows.registry import WorkflowRegistry


NOW = datetime(2026, 9, 10, tzinfo=timezone.utc)


class EchoInput(BaseModel):
    text: str


@dataclass(frozen=True)
class MinimalToolRuntime:
    cancellation: CancellationToken
    execution: ExecutionBudget = field(default_factory=ExecutionBudget)
    deadline: datetime | None = None
    run_awaitable: object = None

    def clock(self):
        return NOW

    def with_timeout(self, seconds):
        deadline = self.clock() + timedelta(seconds=seconds)
        if self.deadline is not None:
            deadline = min(deadline, self.deadline)
        return replace(self, deadline=deadline)

    def raise_if_cancelled(self):
        self.cancellation.raise_if_cancelled()


class ScriptedModel:
    """One native-tool response per model call, including the final answer."""

    def __init__(self, responses=None):
        self.bound_tool_names = []
        self.calls = []
        self.bind_options = []
        self.close_calls = 0
        self.responses = responses if responses is not None else [
            [AIMessage(
                content="", tool_calls=[{
                    "name": "echo_text", "args": {"text": "hello"},
                    "id": "echo-call-1", "type": "tool_call",
                }],
                usage_metadata={"input_tokens": 3, "output_tokens": 1, "total_tokens": 4},
            )],
            [AIMessageChunk(content="hello "), AIMessageChunk(
                content="world",
                usage_metadata={"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            )],
        ]

    def bind_tools(self, tools, **_kwargs):
        self.bound_tool_names = [tool.name for tool in tools]
        self.bind_options.append(_kwargs)
        return self

    def stream(self, messages, **kwargs):
        index = len(self.calls)
        self.calls.append(("model", list(messages)))
        if index >= len(self.responses):
            raise AssertionError("scripted model exhausted: unnecessary model call")
        response = self.responses[index]
        if isinstance(response, Exception):
            raise response
        yield from response

    def close(self):
        self.close_calls += 1


def react_workflow(executor, seen_runtimes, policy=None, middleware=()):
    policy = policy or ReActPolicy()

    def factory(request: RunRequest, context: RunContext, resources: RunResources):
        tool_runtime = MinimalToolRuntime(
            cancellation=context.cancellation,
            execution=ExecutionBudget(
                max_tool_calls=resources.settings.max_tool_calls,
                max_observation_bytes=resources.settings.max_observation_bytes,
            ),
            run_awaitable=resources.async_stream_runner.run,
        )
        seen_runtimes.append(tool_runtime)
        tools = build_langchain_tools(executor.registry.definitions, executor=executor)
        graph = compile_react_graph(
            resources.model, tools,
            stream_chat_message=resources.stream_chat_message, policy=policy, middleware=middleware,
        )
        return WorkflowExecution(
            graph=graph,
            initial_state={
                "messages": build_messages(request.input, context.history or SnapshotHistory()),
                "references": [], "planning_rounds": 0, "tool_call_count": 0,
                "observation_bytes": 0,
                "used_tool_call_ids": [], "stop_reason": None, "final_answer": "",
            },
            context=ReActContext(
                request=request, cancellation=context.cancellation,
                settings=resources.settings, async_stream_runner=resources.async_stream_runner,
                clock=context.clock, tool_runtime=tool_runtime, policy=policy,
            ),
        )

    return WorkflowDefinition("generic_react", factory)


@dataclass(frozen=True)
class GenericEvidence:
    reference_id: str


def make_executor(tool_calls, with_evidence=False):
    def echo(arguments, runtime):
        tool_calls.append(arguments.text)
        if with_evidence:
            return ToolResult(
                content=arguments.text, succeeded=True,
                reference=GenericEvidence(runtime.execution.next_reference_id()),
            )
        return arguments.text

    return ToolExecutor(registry=ToolRegistry([
        ToolSpec(name="echo_text", description="Echo text", input_schema=EchoInput, handler=echo),
    ]))


class AgentRuntimeTest(unittest.TestCase):
    def test_disabled_input_budget_does_not_inject_an_output_token_cap(self):
        settings = AgentConfig(max_input_tokens=None, max_observation_bytes=None, max_plan_model_calls=None)
        events, model, tool_calls, _ = self.run_react(settings=settings)
        self.assertEqual(events[-1].kind, "run.completed")
        self.assertEqual(tool_calls, ["hello"])
        self.assertTrue(model.bind_options)
        self.assertTrue(all("max_tokens" not in options and "max_completion_tokens" not in options
                            for options in model.bind_options))

    def test_unlimited_observation_budget_keeps_consumption_and_enforces_tool_limits(self):
        from agent_core.contracts.errors import AgentLimitReachedError
        from agent_core.planning.react.middleware import _stop_reason

        prior = ExecutionBudget(max_tool_calls=2, max_observation_bytes=100)
        prior.begin_attempt(call_id="old", tool_name="history_read", max_calls_for_tool=None, category="memory")
        prior.commit_result(None, observation_size=90)
        with self.assertRaises(AgentLimitReachedError):
            prior.commit_result(None, observation_size=20)
        restored = ExecutionBudget(max_tool_calls=2, max_observation_bytes=None)
        restored.restore(prior.snapshot())
        self.assertEqual(restored.max_observation_bytes, 100)
        self.assertEqual(restored.stop_reason, "observation_limit")
        with self.assertRaises(AgentLimitReachedError):
            restored.commit_result(None, observation_size=20)
        # Legacy accounting without a saved ceiling may adopt the host's
        # explicit unlimited setting; v2 same-Run limits cannot be relaxed.
        restored = ExecutionBudget(max_tool_calls=2, max_observation_bytes=None)
        restored.restore({**prior.snapshot(), "limits": {}})
        self.assertEqual(restored.observation_bytes, 90)
        self.assertEqual(restored.tool_call_count, 1)
        self.assertIsNone(restored.stop_reason)
        settings = AgentConfig(max_tool_calls=2, max_observation_bytes=None)
        context = SimpleNamespace(settings=settings, tool_runtime=SimpleNamespace(execution=restored))
        self.assertIsNone(_stop_reason({"stop_reason": "observation_limit", "observation_bytes": 1_000_000}, context))
        restored.begin_attempt(call_id="new", tool_name="history_read", max_calls_for_tool=None, category="memory")
        restored.commit_result(None, observation_size=1_000_000)
        self.assertEqual(restored.public_budget()["observationBytes"], 1_000_090)
        self.assertIsNone(restored.public_budget()["maxObservationBytes"])
        with self.assertRaises(AgentLimitReachedError):
            restored.begin_attempt(call_id="third", tool_name="history_read", max_calls_for_tool=None, category="memory")
        self.assertEqual(_stop_reason({}, context), "tool_call_limit")

    def test_native_observation_limit_retains_reason_and_does_not_publish_rejected_content(self):
        from agent_core.tools.native import NativeToolGovernance

        budget = ExecutionBudget(max_tool_calls=2, max_observation_bytes=106)
        emitted = []
        runtime = SimpleNamespace(context=SimpleNamespace(cancellation=CancellationToken(),
            tool_runtime=SimpleNamespace(execution=budget)), stream_writer=emitted.append,
            state={"activity_scope_id": "native-stage", "activity_label": "计划执行"})
        request = SimpleNamespace(runtime=runtime, tool=SimpleNamespace(extras={}),
            tool_call={"id": "read", "name": "history_read", "args": {}})
        result = NativeToolGovernance(metadata={"history_read": {"budget_group": "memory"}},
            authorizer=lambda **_: AuthorizationDecision(True)).execute(
            request, lambda _: ToolMessage(content="private rejected evidence" * 100,
                                           tool_call_id="read", name="history_read"))
        receipt = json.loads(result.update["messages"][0].content)
        self.assertEqual(receipt["errorCode"], "observation_limit")
        self.assertEqual(budget.observation_bytes, 0)
        self.assertEqual(budget.tool_call_count, 1)
        self.assertEqual(budget.completed_call_ids, {"read"})
        completed = next(event for event in emitted if event["kind"] == "tool.completed")
        self.assertEqual(completed["errorCode"], "observation_limit")
        self.assertNotIn("private rejected evidence", str(completed))
        self.assertTrue(all(event["activityScopeId"] == "native-stage" and event["activityLabel"] == "计划执行"
                            for event in emitted))

    def test_tool_receipts_recovered_from_updates_publish_only_known_error_codes(self):
        receipts = [ToolMessage(content=json.dumps({"ok": False, "errorCode": code,
                    "message": "private backend details"}), status="error", name="read_file", tool_call_id=str(i))
                    for i, code in enumerate(("file_not_found", "observation_limit", "unknown private code"))]

        class Graph:
            async def astream(self, *_args, **_kwargs):
                yield {"type": "custom", "data": {"kind": "tool.planned", "callId": "0", "stepId": "tool_1",
                    "activityScopeId": "planned-stage", "activityLabel": "计划执行", "toolName": "read_file"}}
                yield {"type": "updates", "data": {"restored_stage": {"messages": receipts}}}
                yield {"type": "updates", "data": {"replayed_stage": {"messages": receipts}}}
                yield {"type": "values", "data": {"final_answer": "Not completed"}}

        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition(
            "receipts", lambda *_: WorkflowExecution(Graph(), {"activity_scope_id": "restored-stage",
                "activity_label": "执行结果核对"}))]), model_factory=ScriptedModel)
        events = list(service.stream(RunRequest("run", "receipts", ""),
                                     RunContext(cancellation=CancellationToken())))
        observed = [event for event in events if event.kind == "tool.observed"]
        self.assertEqual([event.payload["errorCode"] for event in observed],
                         ["file_not_found", "observation_limit", "tool_query_failed"])
        self.assertTrue(all(event.payload["status"] == "failed" for event in observed))
        self.assertEqual([event.activity_scope_id for event in observed], ["planned-stage", "restored-stage", "restored-stage"])
        self.assertEqual([event.activity_label for event in observed], ["计划执行", "执行结果核对", "执行结果核对"])
        self.assertEqual(observed[0].step_id, "tool_1")
        self.assertNotIn("private", str(observed))

    def run_react(self, observer=None, history=None, with_evidence=False, *,
                  model=None, settings=None, policy=None, authorizer=None, middleware=()):
        model = model or ScriptedModel()
        tool_calls, runtimes = [], []
        executor = make_executor(tool_calls, with_evidence=with_evidence)
        if authorizer is not None:
            executor.authorizer = authorizer
        service = AgentService(
            workflows=WorkflowRegistry([react_workflow(executor, runtimes, policy, middleware)]),
            model_factory=lambda: model, settings=settings or AgentConfig(), usage_observer=observer,
        )
        request = RunRequest("plain-run", "generic_react", "Use the echo tool", attempt=3)
        context = RunContext(
            cancellation=CancellationToken(), history=history or SnapshotHistory(), clock=lambda: NOW,
        )
        events = list(service.stream(request, context))
        return events, model, tool_calls, runtimes

    def test_uncited_tool_completes_actual_react_and_preserves_run_identifiers(self):
        events, model, tool_calls, runtimes = self.run_react()
        self.assertEqual(model.bound_tool_names, ["echo_text"])
        self.assertEqual(tool_calls, ["hello"])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(events[-1].kind, "run.completed")
        self.assertEqual(events[-1].payload["result"].output, "hello world")
        self.assertFalse(any(event.kind == "reference.added" for event in events))
        self.assertEqual(runtimes[0].execution.references_snapshot(), ())
        self.assertEqual(runtimes[0].execution.observation_bytes, 5)
        self.assertTrue(all(event.run_id == "plain-run" and event.attempt == 3 for event in events))
        self.assertEqual([event.sequence for event in events], list(range(1, len(events) + 1)))
        tool_events = [event for event in events if event.kind.startswith("tool.")]
        self.assertTrue(tool_events)
        self.assertTrue(all(event.call_id == "echo-call-1" and event.step_id == "tool_1" for event in tool_events))
        self.assertTrue(all(event.activity_scope_id == "run" and event.activity_label == "任务执行" for event in tool_events))
        update = tool_events[0].as_update()
        self.assertEqual((update["callId"], update["stepId"], update["activityScopeId"], update["activityLabel"]),
                         ("echo-call-1", "tool_1", "run", "任务执行"))
        self.assertEqual(model.close_calls, 1)

    def test_native_langchain_chat_model_uses_tool_binding_and_finishes_after_observation(self):
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from pydantic import Field

        class NativeToolModel(FakeMessagesListChatModel):
            generated_calls: int = 0
            seen_tools: list[str] = Field(default_factory=list)

            def bind_tools(self, tools, **_kwargs):
                self.seen_tools = [tool.name for tool in tools]
                return self

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                self.generated_calls += 1
                if self.generated_calls == 2:
                    if not any(isinstance(message, ToolMessage) and
                               message.content == "native" for message in messages):
                        raise AssertionError("native model did not receive the tool observation")
                return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

        model = NativeToolModel(responses=[
            AIMessage(content="", tool_calls=[{
                "name": "echo_text", "args": {"text": "native"},
                "id": "native-call", "type": "tool_call",
            }]),
            AIMessage(content="Native model answer"),
        ])
        events, _, tool_calls, _ = self.run_react(model=model)
        self.assertEqual(model.seen_tools, ["echo_text"])
        self.assertEqual(model.generated_calls, 2)
        self.assertEqual(tool_calls, ["native"])
        self.assertEqual(events[-1].payload["result"].output, "Native model answer")

    def test_public_langchain_middleware_can_extend_the_model_input(self):
        from langchain.agents.middleware import AgentMiddleware

        class AddContext(AgentMiddleware):
            def before_model(self, state, runtime):
                return {"messages": [HumanMessage(content="Context supplied by public middleware")]}

            async def awrap_model_call(self, request, handler):
                return await handler(request.override(model_settings={"temperature": 0.25}))

        model = ScriptedModel(responses=[[AIMessageChunk(content="Extended answer")]])
        events, _, _, _ = self.run_react(model=model, middleware=[AddContext()])
        self.assertEqual(len(model.calls), 1)
        self.assertIn("Context supplied by public middleware", [message.content
                      for message in model.calls[0][1]])
        self.assertEqual(model.bind_options, [{"temperature": 0.25, "max_tokens": 4096}])
        self.assertEqual(events[-1].payload["result"].output, "Extended answer")

    def test_public_middleware_answer_rewrite_is_validated_and_published(self):
        from langchain.agents.middleware import AgentMiddleware

        class RewriteAnswer(AgentMiddleware):
            def after_model(self, state, runtime):
                return {"messages": [state["messages"][-1].model_copy(
                    update={"content": "Answer revised by public middleware"},
                )]}

        class RecordValidation(ReActPolicy):
            def __init__(self):
                self.answers = []

            def validate_output(self, answer, references):
                self.answers.append(answer)

        policy = RecordValidation()
        model = ScriptedModel(responses=[[AIMessageChunk(content="Original provider candidate")]])
        events, _, _, _ = self.run_react(model=model, policy=policy, middleware=[RewriteAnswer()])
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(events[-1].payload["result"].output, "Answer revised by public middleware")
        public_deltas = "".join(event.payload["payload"]["delta"] for event in events
                                if event.kind == "answer.delta")
        self.assertEqual(public_deltas, "Answer revised by public middleware")
        self.assertTrue(policy.answers)
        self.assertTrue(all(answer == "Answer revised by public middleware" for answer in policy.answers))
        self.assertNotIn("Original provider candidate", repr(events))

    def test_model_without_implemented_tool_binding_has_stable_error_and_is_closed(self):
        class UnsupportedToolModel(ScriptedModel):
            def bind_tools(self, tools, **kwargs):
                raise self.binding_error

        for error in (NotImplementedError("private binding failure"),
                      AttributeError("private binding failure")):
            with self.subTest(error_type=type(error).__name__):
                model = UnsupportedToolModel()
                model.binding_error = error
                with self.assertRaises(ToolCallingUnsupportedError) as raised:
                    self.run_react(model=model)
                self.assertNotIn("private binding failure", str(raised.exception))
                self.assertEqual(model.calls, [])
                self.assertEqual(model.close_calls, 1)

    def test_community_model_retry_records_each_provider_attempt_once(self):
        from langchain.agents.middleware import ModelRetryMiddleware

        model = ScriptedModel(responses=[
            RuntimeError("temporary provider failure"),
            [AIMessageChunk(content="Recovered answer", usage_metadata={
                "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
            })],
        ])
        observer = InMemoryUsageObserver()
        events, _, tool_calls, _ = self.run_react(
            model=model, observer=observer,
            middleware=[ModelRetryMiddleware(
                max_retries=1, initial_delay=0, jitter=False, on_failure="error",
            )],
        )
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(tool_calls, [])
        self.assertEqual(events[-1].payload["result"].output, "Recovered answer")
        self.assertEqual(len(observer.records), 2)
        self.assertEqual(len({record.call_id for record in observer.records}), 2)
        self.assertEqual([record.status for record in observer.records], ["failed", "succeeded"])
        self.assertEqual([record.total_tokens for record in observer.records], [None, 3])
        self.assertEqual([record.phase for record in observer.records], ["decision", "decision"])
        self.assertEqual(model.close_calls, 1)

    def test_no_tool_answer_completes_in_one_model_call(self):
        model = ScriptedModel(responses=[[AIMessageChunk(content="Direct answer")]])
        observer = InMemoryUsageObserver()
        events, _, tool_calls, _ = self.run_react(model=model, observer=observer)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(tool_calls, [])
        self.assertEqual(events[-1].payload["result"].output, "Direct answer")
        self.assertFalse(any(event.kind.startswith("tool.") for event in events))
        self.assertEqual(len(observer.records), 1)
        self.assertIsNone(observer.records[0].total_tokens)

    def test_tool_turn_text_is_not_published_as_answer_when_tool_call_arrives_late(self):
        model = ScriptedModel(responses=[
            [AIMessageChunk(content="private intermediate draft"), AIMessageChunk(
                content="", tool_call_chunks=[{
                    "name": "echo_text", "args": '{"text":"hello"}',
                    "id": "echo-call-1", "index": 0,
                }],
            )],
            [AIMessageChunk(content="Public result")],
        ])
        events, _, tool_calls, _ = self.run_react(model=model)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(tool_calls, ["hello"])
        self.assertEqual(events[-1].payload["result"].output, "Public result")
        public_deltas = "".join(event.payload["payload"]["delta"] for event in events
                                if event.kind == "answer.delta")
        self.assertEqual(public_deltas, "Public result")
        self.assertNotIn("private intermediate draft", repr(events))

    def test_tool_denial_is_observed_and_counted_without_calling_handler(self):
        class DenyTools:
            def authorize(self, definition, runtime):
                return AuthorizationDecision(False, "denied_for_test", "Tool access denied")

        events, model, tool_calls, runtimes = self.run_react(authorizer=DenyTools())
        self.assertEqual(tool_calls, [])
        self.assertEqual(runtimes[0].execution.tool_call_count, 1)
        self.assertEqual(runtimes[0].execution.observation_bytes, 0)
        observation = next(message for message in model.calls[1][1]
                           if isinstance(message, ToolMessage))
        self.assertEqual(observation.status, "error")
        # Unauthorized tools are removed before binding; guessed calls fail at
        # the shared visibility boundary without invoking their handlers.
        self.assertEqual(json.loads(str(observation.content))["errorCode"], "tool_not_exposed")
        self.assertTrue(any(event.kind == "tool.completed" and
                            event.payload["status"] == "failed" for event in events))
        self.assertEqual(events[-1].kind, "run.completed")

    def test_batch_exceeding_budget_performs_no_tools_and_only_one_limited_answer(self):
        model = ScriptedModel(responses=[
            [AIMessage(content="", additional_kwargs={"tool_calls": [
                {"id": "batch-one", "type": "function", "function": {
                    "name": "echo_text", "arguments": '{"text":"RAW_REJECTED_BATCH"}',
                }},
                {"id": "batch-two", "type": "function", "function": {
                    "name": "echo_text", "arguments": '{"text":"two"}',
                }},
            ]}, tool_calls=[
                {"name": "echo_text", "args": {"text": "one"},
                 "id": "batch-one", "type": "tool_call"},
                {"name": "echo_text", "args": {"text": "two"},
                 "id": "batch-two", "type": "tool_call"},
            ])],
            [AIMessageChunk(content="Insufficient execution budget")],
        ])
        observer = InMemoryUsageObserver()
        events, _, tool_calls, runtimes = self.run_react(
            model=model, settings=AgentConfig(max_tool_calls=1), observer=observer,
        )
        self.assertEqual(tool_calls, [])
        self.assertEqual(runtimes[0].execution.tool_call_count, 0)
        self.assertEqual(len(model.calls), 2)
        summary_history = model.calls[1][1]
        rejected_messages = [message for message in summary_history if isinstance(message, AIMessage)]
        self.assertEqual(len(rejected_messages), 1)
        self.assertEqual(rejected_messages[0].tool_calls, [])
        self.assertEqual(rejected_messages[0].invalid_tool_calls, [])
        self.assertNotIn("tool_calls", rejected_messages[0].additional_kwargs)
        self.assertNotIn("RAW_REJECTED_BATCH", repr(summary_history))
        self.assertNotIn("batch-one", repr(summary_history))
        self.assertNotIn("batch-two", repr(summary_history))
        self.assertEqual([record.phase for record in observer.records], ["decision", "answer"])
        self.assertEqual(
            events[-1].payload["result"].output,
            "已达到本轮执行上限，以下回答基于已获得的信息。\n\nInsufficient execution budget",
        )

    def test_policy_rejects_final_candidate_and_still_closes_model(self):
        class RejectAnswer(ReActPolicy):
            def validate_output(self, answer, references):
                raise OutputValidationError("Output rejected by application policy")

        model = ScriptedModel(responses=[[AIMessageChunk(content="Candidate answer")]])
        with self.assertRaisesRegex(OutputValidationError, "Output rejected"):
            self.run_react(model=model, policy=RejectAnswer())
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.close_calls, 1)

    def test_registering_fixed_graph_reuses_service_without_executor_changes(self):
        class FixedState(TypedDict):
            value: int

        def fixed_factory(request, context, resources):
            builder = StateGraph(FixedState)
            builder.add_node("increment", lambda state: {"value": state["value"] + 1})
            builder.add_edge(START, "increment")
            builder.add_edge("increment", END)
            return WorkflowExecution(
                graph=builder.compile(), initial_state={"value": int(request.input)},
                result_factory=lambda state: RunResult(output=state["value"]),
            )

        tool_calls, runtimes, models = [], [], []
        executor = make_executor(tool_calls)

        def model_factory():
            model = ScriptedModel()
            models.append(model)
            return model

        registry = WorkflowRegistry([react_workflow(executor, runtimes)])
        service = AgentService(workflows=registry, model_factory=model_factory)
        registry.register(WorkflowDefinition("fixed_increment", fixed_factory))
        fixed = list(service.stream(
            RunRequest("fixed", "fixed_increment", "41"),
            RunContext(cancellation=CancellationToken()),
        ))
        self.assertEqual(fixed[-1].payload["result"].output, 42)
        self.assertEqual(tool_calls, [])
        self.assertEqual(models[0].calls, [])
        react = list(service.stream(
            RunRequest("react", "generic_react", "Use echo"),
            RunContext(cancellation=CancellationToken()),
        ))
        self.assertEqual(react[-1].payload["result"].output, "hello world")
        self.assertEqual(tool_calls, ["hello"])
        self.assertEqual(registry.names, ("generic_react", "fixed_increment"))

    def test_optional_evidence_event_retains_tool_call_and_step_identifiers(self):
        events, _, _, runtimes = self.run_react(with_evidence=True)
        evidence_events = [event for event in events if event.kind == "reference.added"]
        self.assertEqual(len(evidence_events), 1)
        event = evidence_events[0]
        self.assertEqual((event.call_id, event.step_id), ("echo-call-1", "tool_1"))
        self.assertEqual(event.payload["payload"], {"reference_id": "R1"})
        self.assertEqual(len(runtimes[0].execution.references_snapshot()), 1)
        self.assertEqual(events[-1].payload["result"].output, "hello world")

    def test_replacing_history_reader_changes_actual_model_input(self):
        class Reader:
            def __init__(self):
                self.read_calls = 0

            def read(self):
                self.read_calls += 1
                return [HistoryMessage("new question", "new answer")]

        _, original, _, _ = self.run_react(history=SnapshotHistory([
            HistoryMessage("old question", "old answer"),
        ]))
        reader = Reader()
        _, changed, _, _ = self.run_react(history=reader)
        first_input = [str(item.content) for item in original.calls[0][1]]
        changed_input = [str(item.content) for item in changed.calls[0][1]]
        self.assertIn("old question", first_input)
        self.assertNotIn("old question", changed_input)
        self.assertIn("new question", changed_input)
        self.assertIn("new answer", changed_input)
        self.assertEqual(reader.read_calls, 1)
        self.assertIsInstance(changed.calls[0][1][-1], HumanMessage)
        self.assertEqual(changed.calls[0][1][-1].content, "Use the echo tool")

    def test_default_workflow_result_does_not_publish_private_graph_state(self):
        from agent_core.workflows.contracts import default_result

        result = default_result({"messages": ["private prompt"], "scratch": "internal"})
        self.assertIsNone(result.output)
        self.assertEqual(result.artifacts, ())

    def test_graph_close_failure_preserves_original_run_error_and_closes_model(self):
        from agent_core.contracts.errors import OutputValidationError, ToolCallingUnsupportedError

        class GraphStream:
            def __aiter__(self):
                return self

            async def __anext__(self):
                raise OutputValidationError("original output failure")

            async def aclose(self):
                raise RuntimeError("cleanup failure")

        class Graph:
            def astream(self, *_args, **_kwargs):
                return GraphStream()

        model = ScriptedModel()
        service = AgentService(
            workflows=WorkflowRegistry([WorkflowDefinition(
                "broken_graph", lambda *_: WorkflowExecution(Graph(), {}),
            )]), model_factory=lambda: model,
        )
        with self.assertRaisesRegex(OutputValidationError, "original output failure"):
            list(service.stream(RunRequest("error-run", "broken_graph", ""),
                                RunContext(cancellation=CancellationToken())))
        self.assertEqual(model.close_calls, 1)

    def test_usage_observer_is_optional_and_cannot_change_execution_events(self):
        class FailingObserver:
            def __init__(self):
                self.calls = 0

            def record(self, usage):
                self.calls += 1
                raise RuntimeError("observer unavailable")

        in_memory = InMemoryUsageObserver()
        failing = FailingObserver()
        outputs = []
        for observer in (None, NoopUsageObserver(), in_memory, failing):
            events, model, tool_calls, _ = self.run_react(observer=observer)
            outputs.append([asdict(event) for event in events])
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(tool_calls, ["hello"])
        self.assertTrue(all(output == outputs[0] for output in outputs[1:]))
        self.assertEqual(failing.calls, 2)
        records = in_memory.records
        self.assertEqual(len(records), 2)
        self.assertEqual(len({record.call_id for record in records}), 2)
        self.assertTrue(all(record.run_id == "plain-run" and record.attempt == 3 for record in records))
        self.assertEqual([record.phase for record in records], ["decision", "decision"])
        self.assertEqual([record.step_id for record in records],
                         ["agent_decision_1", "agent_decision_2"])
        self.assertEqual([record.total_tokens for record in records], [4, 7])
        self.assertTrue(all(record.status == "succeeded" and record.source == "reported" for record in records))
