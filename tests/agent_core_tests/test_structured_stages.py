"""Real create_agent response-format routing through the owned ReAct hooks."""

from dataclasses import replace
import unittest
from types import SimpleNamespace
from typing import TypedDict

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from agent_core.config import AgentConfig
from agent_core.contracts.context import RunContext
from agent_core.contracts.errors import AgentContextError, AgentUpstreamError
from agent_core.contracts.runs import RunRequest, RunResult
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.service import AgentService
from agent_core.workflows.contracts import WorkflowDefinition, WorkflowExecution
from agent_core.workflows.registry import WorkflowRegistry
from agent_core_tests.test_runtime import MinimalToolRuntime, ScriptedModel


class StageResult(BaseModel):
    count: int


class BudgetProbe:
    def __init__(self):
        self.calls = 0
        self.usage = []

    def begin_call(self):
        self.calls += 1

    def record_usage(self, message):
        self.usage.append(message)

    def snapshot(self):
        return {"revision": self.calls, "model_calls": self.calls}

    def restore(self, value):
        self.calls = max(self.calls, value.get("model_calls", 0))


class StructuredStageTests(unittest.TestCase):
    def test_cancelled_workflow_saves_consumed_allowance_before_error_propagates(self):
        from agent_core.contracts.errors import AgentCancelled
        from langgraph.checkpoint.memory import InMemorySaver
        from langgraph.graph import END, START, StateGraph
        from langgraph.runtime import Runtime

        class MeterState(TypedDict):
            calls: int

        saver = InMemorySaver()
        meter = SimpleNamespace(calls=0)
        finalized = []

        def fail(state, runtime: Runtime):
            runtime.context.calls += 1
            raise AgentCancelled("cancelled during a call")

        def finalize(graph, config, context, error):
            graph.update_state(config, {"calls": context.calls})
            finalized.append(type(error))

        def factory(request, context, resources):
            builder = StateGraph(MeterState)
            builder.add_node("fail", fail)
            builder.add_edge(START, "fail")
            builder.add_edge("fail", END)
            return WorkflowExecution(builder.compile(checkpointer=saver), {"calls": 0},
                                     meter, finalize_error=finalize)

        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("cancel", factory)]))
        events = []
        with self.assertRaises(AgentCancelled):
            for event in service.stream(RunRequest("cancel-run", "cancel", "Run"),
                                        RunContext(CancellationToken())):
                events.append(event)
        self.assertEqual(finalized, [AgentCancelled])
        saved = saver.get_tuple({"configurable": {"thread_id": "cancel-run"}})
        self.assertEqual(saved.checkpoint["channel_values"]["calls"], 1)
        self.assertFalse(any(event.kind == "run.completed" for event in events))

    def test_rubric_feedback_is_not_archived_as_user_provenance(self):
        from agent_core.context.budget import archive_messages
        entries = []
        binding = SimpleNamespace(scope=SimpleNamespace(run_id="run"),
            history=SimpleNamespace(append=lambda scope, value: entries.extend(value)))
        archive_messages(binding, [HumanMessage(content="My request", id="user"),
            HumanMessage(content="Fix the output", id="feedback", name="rubric_grader",
                additional_kwargs={"lc_source": "rubric_grader"})])
        self.assertEqual([entry.role for entry in entries], ["user", "assistant"])
        self.assertEqual(entries[1].metadata["internal_source"], "rubric_grader")

    def run_stage(self, responses, *, settings=None, schema=StageResult, model=None, probe=None,
                  with_tools=False, policy=None, saver=None, execution=None, tools=()):
        model = model or ScriptedModel(responses)
        probe = probe or BudgetProbe()
        seen = []

        def factory(request, context, resources):
            from agent_core.context import build_messages
            runtime = MinimalToolRuntime(context.cancellation, execution=execution) if execution is not None else MinimalToolRuntime(context.cancellation)
            seen.append(runtime)
            stage_policy = policy or ReActPolicy()
            graph = compile_react_graph(
                model, list(tools), stream_chat_message=resources.stream_chat_message,
                policy=stage_policy, response_format=ToolStrategy(schema), publish_final=False,
                system_prompt="Return the stage result.",
                middleware=[TodoListMiddleware()] if with_tools else (), checkpointer=saver,
            )
            return WorkflowExecution(graph, {
                "messages": build_messages(request.input, context.history or ()), "references": [], "planning_rounds": 0,
                "tool_call_count": 0, "observation_bytes": 0, "used_tool_call_ids": [],
            }, ReActContext(request, context.cancellation, resources.settings,
                            resources.async_stream_runner, context.clock, runtime, stage_policy,
                            model_budget=probe),
                result_factory=lambda state: RunResult(state.get("structured_response"),
                    metadata={"ledger": state.get("model_ledger")}))

        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("stage", factory)]),
                               model_factory=lambda: model, settings=settings)
        events = list(service.stream(RunRequest("stage-run", "stage", "Count"),
                                     RunContext(CancellationToken())))
        return events, model, probe, seen[0]

    def test_schema_repair_counts_models_not_tool_io_and_does_not_publish_answer(self):
        messages = [[AIMessage(content="", tool_calls=[{
            "name": "StageResult", "args": {"count": value}, "id": f"schema-{index}",
            "type": "tool_call",
        }])] for index, value in enumerate(["not an integer", 7])]
        events, model, probe, runtime = self.run_stage(messages)
        self.assertEqual(events[-1].payload["result"].output.count, 7)
        self.assertEqual(probe.calls, 2)
        self.assertEqual(len(probe.usage), 2)
        self.assertEqual(runtime.execution.tool_call_count, 0)
        self.assertEqual(events[-1].payload["result"].metadata["ledger"]["model_calls"], 2)
        self.assertFalse(any(event.kind == "answer.delta" for event in events))
        self.assertFalse(any(event.kind == "tool.observed" for event in events))
        self.assertIn("StageResult", model.bound_tool_names)

    def test_response_schema_is_in_complete_input_budget(self):
        from pydantic import Field, create_model
        huge = create_model("HugeStage", detail=(str, Field(description="large schema " * 1000)))
        with self.assertRaises(AgentContextError):
            self.run_stage([], settings=replace(AgentConfig(), max_input_tokens=500), schema=huge)

    def test_business_tool_limit_still_allows_bounded_structured_finalization(self):
        from agent_core.runtime.budget import ExecutionBudget
        from agent_core.contracts.errors import AgentLimitReachedError
        execution = ExecutionBudget()
        execution.begin_attempt(call_id="used", tool_name="query", max_calls_for_tool=1)
        with self.assertRaises(AgentLimitReachedError) as caught:
            execution.begin_attempt(call_id="denied", tool_name="query", max_calls_for_tool=1)
        self.assertIn("query", caught.exception.public_message)
        self.assertIn("1/1", caught.exception.public_message)
        valid = AIMessage(content="", tool_calls=[{"name": "StageResult", "args": {"count": 7}, "id": "result", "type": "tool_call"}])
        events, model, probe, runtime = self.run_stage([[AIMessage(content="Only existing evidence")], [valid]],
                                                      execution=execution, with_tools=True)
        self.assertEqual(events[-1].payload["result"].output.count, 7)
        self.assertEqual(probe.calls, 2)
        # The denied logical call is charged globally, but only query is spent.
        self.assertEqual(runtime.execution.tool_call_count, 2)
        self.assertEqual(runtime.execution.calls_by_tool, {"query": 1})
        self.assertIsNone(runtime.execution.stop_reason)
        self.assertIn("write_todos", model.bound_tool_names)
        self.assertIn("StageResult", model.bound_tool_names)
        self.assertIn("工具 query", model.calls[-1][1][0].text)

    def test_spent_tool_is_hidden_before_another_attempt_and_allowance_is_visible(self):
        from agent_core.runtime.budget import ExecutionBudget
        from agent_core.tools import ToolExecutor, ToolRegistry, ToolSpec
        from agent_core.tools.langchain import build_langchain_tools
        execution = ExecutionBudget()
        execution.begin_attempt(call_id="used", tool_name="query", max_calls_for_tool=1)
        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(name="query", description="Read count", input_schema=StageResult,
            max_calls_per_run=1, handler=lambda *_: self.fail("spent tool must not execute"))]))
        tools = build_langchain_tools(executor.registry.definitions, executor=executor)
        valid = AIMessage(content="", tool_calls=[{"name": "StageResult", "args": {"count": 7}, "id": "result", "type": "tool_call"}])
        _, model, probe, runtime = self.run_stage([[valid]], execution=execution, tools=tools)
        self.assertNotIn("query", model.bound_tool_names)
        self.assertEqual(runtime.execution.tool_call_count, 1)
        self.assertIsNone(runtime.execution.stop_reason)
        self.assertEqual(probe.calls, 1)
        self.assertIn('"remaining": 0', model.calls[0][1][0].text)
        self.assertEqual(execution.public_budget()["toolLimits"]["query"], {"used": 1, "limit": 1, "remaining": 0})
        restored = ExecutionBudget(max_tool_calls=2)
        restored.restore(execution.snapshot())
        self.assertEqual(restored.tool_call_count, 1)
        self.assertEqual(restored.max_tool_calls, 2)

    def test_exhausted_query_category_leaves_memory_and_output_tools_available(self):
        from agent_core.runtime.budget import ExecutionBudget
        from agent_core.tools import ToolExecutor, ToolRegistry, ToolSpec
        from agent_core.tools.langchain import build_langchain_tools
        execution = ExecutionBudget()
        execution.begin_attempt(call_id="used", tool_name="query", max_calls_for_tool=8)
        specs = [ToolSpec(name=name, description="Read count", input_schema=StageResult, category=category,
            budget_group="memory" if category == "memory" else "external",
            handler=lambda *_: self.fail("no service call expected")) for name, category in [("query", "general"), ("recall", "memory")]]
        executor = ToolExecutor(registry=ToolRegistry(specs))
        valid = AIMessage(content="", tool_calls=[{"name": "StageResult", "args": {"count": 7}, "id": "result", "type": "tool_call"}])
        _, model, _, _ = self.run_stage([[valid]], execution=execution,
            settings=replace(AgentConfig(), max_external_tool_calls=1),
            tools=build_langchain_tools(executor.registry.definitions, executor=executor))
        self.assertNotIn("query", model.bound_tool_names)
        self.assertIn("recall", model.bound_tool_names)
        self.assertIn("StageResult", model.bound_tool_names)
        self.assertIsNone(execution.stop_reason)

    def test_plain_text_is_not_structured_output_and_repairs_are_bounded(self):
        model = ScriptedModel([[AIMessage(content='{"count": 7}')]] * 3)
        probe = BudgetProbe()
        with self.assertRaisesRegex(AgentUpstreamError, "格式修复上限"):
            self.run_stage([], model=model, probe=probe, with_tools=True)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(probe.calls, 3)
        self.assertEqual(len(probe.usage), 3)
        self.assertTrue(all(options["tool_choice"] == "any" for options in model.bind_options))
        self.assertTrue(all("前次未取得有效" in call[1][0].text for call in model.calls[1:]))

    def test_truncated_schema_tool_arguments_retry_without_business_tool_dispatch(self):
        invalid = AIMessage(content="", invalid_tool_calls=[{
            "name": "StageResult", "args": '{"count":', "id": "broken", "error": "invalid JSON", "type": "invalid_tool_call",
        }], response_metadata={"finish_reason": "length"})
        valid = AIMessage(content="", tool_calls=[{"name": "StageResult", "args": {"count": 7}, "id": "fixed", "type": "tool_call"}])
        events, model, probe, runtime = self.run_stage([[invalid], [valid]], with_tools=True)
        self.assertEqual(events[-1].payload["result"].output.count, 7)
        self.assertEqual(probe.calls, 2)
        self.assertEqual(runtime.execution.tool_call_count, 0)
        self.assertFalse(any(getattr(message, "invalid_tool_calls", []) for message in model.calls[1][1]))
        self.assertFalse(any(event.kind == "tool.observed" for event in events))
        self.assertEqual([message.content for message in model.calls[1][1] if isinstance(message, HumanMessage)], ["Count"])

    def test_repeated_length_terminated_empty_output_reports_truncation(self):
        model = ScriptedModel([[AIMessage(content="", response_metadata={"finish_reason": "length"})]] * 3)
        probe = BudgetProbe()
        with self.assertRaisesRegex(AgentUpstreamError, "输出达到长度上限"):
            self.run_stage([], model=model, probe=probe, with_tools=True)
        self.assertEqual(probe.calls, 3)

    def test_schema_repairs_cannot_reset_a_preconsumed_plan_budget(self):
        from agent_core.contracts.errors import AgentLimitReachedError
        from agent_core.planning.plan_execute.budget import PlanModelBudget
        from langgraph.checkpoint.memory import InMemorySaver

        probe = PlanModelBudget(max_calls=2)
        probe.begin_call()
        probe.record_usage(None)
        model = ScriptedModel([[AIMessage(content="No structured output")]] * 3)
        saver = InMemorySaver()
        with self.assertRaises(AgentLimitReachedError) as caught:
            self.run_stage([], model=model, probe=probe, with_tools=True, saver=saver)
        self.assertEqual(caught.exception.code, "agent_limit_reached")
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(probe.calls, 2)
        self.assertEqual(probe.recorded_calls, 2)
        saved = saver.get_tuple({"configurable": {"thread_id": "stage-run"}})
        self.assertEqual(saved.checkpoint["channel_values"]["structured_output_retries"], 1)
        self.assertEqual(saved.checkpoint["channel_values"]["model_ledger"]["calls"], 2)

    def test_structured_policy_hook_replaces_ordinary_answer_prompts(self):
        class StructuredPolicy(ReActPolicy):
            def planner_prompt(self, *_args):
                raise AssertionError("Ordinary planner prompt must not override a structured stage")

            def final_prompt(self, *_args):
                raise AssertionError("Ordinary final-answer prompt must not override a structured stage")

            def structured_prompt(self, context, tool_names, output_tool_names):
                assert "write_todos" in tool_names and output_tool_names == ["StageResult"]
                return "Preserve domain-specific read constraints."

        valid = AIMessage(content="", tool_calls=[{"name": "StageResult", "args": {"count": 7}, "id": "fixed", "type": "tool_call"}])
        _, model, _, _ = self.run_stage([[valid]], policy=StructuredPolicy(), with_tools=True)
        self.assertIn("Preserve domain-specific read constraints", model.calls[0][1][0].text)
        self.assertIn("输出工具只提交阶段数据", model.calls[0][1][0].text)

    def test_real_openai_adapter_repairs_empty_length_response_and_preserves_schema_wire_contract(self):
        import asyncio
        import json
        import httpx
        from langgraph.checkpoint.memory import InMemorySaver
        from unittest.mock import patch
        from agent_core.models.factory import build_chat_model
        from agent_core.models.profiles import ModelProfile

        # SDK client headers probe platform.platform() in a background thread;
        # fix this unrelated metadata so the no-subprocess audit stays strict.
        self.enterContext(patch("platform.platform", return_value="Linux-fixture"))
        wire = []
        def provider(request):
            wire.append(json.loads(request.content))
            if len(wire) == 1:
                delta, finish = {"role": "assistant", "content": ""}, "length"
            else:
                delta, finish = {"role": "assistant", "tool_calls": [{"index": 0, "id": "schema-fixed", "type": "function",
                    "function": {"name": "StageResult", "arguments": '{"count":7}'}}]}, "tool_calls"
            chunks = [
                {"id": "fixture", "object": "chat.completion.chunk", "model": "fixture-model",
                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                {"id": "fixture", "object": "chat.completion.chunk", "model": "fixture-model",
                 "choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                 "usage": {"prompt_tokens": 12384 if len(wire) == 1 else 10,
                           "completion_tokens": 4096 if len(wire) == 1 else 5,
                           "total_tokens": 16480 if len(wire) == 1 else 15,
                           "completion_tokens_details": {"reasoning_tokens": 4081 if len(wire) == 1 else 0}}},
            ]
            data = "".join("data: " + json.dumps(chunk) + "\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"}, request=request)

        class OfflineClient(httpx.Client):
            def __init__(self, **kwargs):
                super().__init__(transport=httpx.MockTransport(provider), **kwargs)

        class OfflineAsyncClient(httpx.AsyncClient):
            def __init__(self, **kwargs):
                super().__init__(transport=httpx.MockTransport(provider), **kwargs)

        profile = ModelProfile(model="glm-5.3-flash", api_key="fixture-not-a-secret",
                               base_url="https://open.bigmodel.cn/api/paas/v4/", max_tokens=4096)
        with patch("httpx.Client", OfflineClient), patch("httpx.AsyncClient", OfflineAsyncClient):
            model = build_chat_model(profile)
        self.addCleanup(model.close)
        self.addCleanup(lambda: asyncio.run(model.aclose()))
        saver = InMemorySaver()
        events, _, probe, runtime = self.run_stage([], model=model, with_tools=True, saver=saver)
        self.assertEqual(events[-1].payload["result"].output.count, 7)
        self.assertEqual(probe.calls, 2)
        self.assertEqual(probe.usage[0].response_metadata["finish_reason"], "length")
        self.assertEqual(probe.usage[0].text, "")
        self.assertEqual(probe.usage[0].tool_calls, [])
        self.assertEqual(probe.usage[0].invalid_tool_calls, [])
        self.assertEqual(probe.usage[0].usage_metadata["input_tokens"], 12384)
        self.assertEqual(probe.usage[0].usage_metadata["output_tokens"], 4096)
        self.assertEqual(probe.usage[0].usage_metadata["total_tokens"], 16480)
        self.assertEqual(probe.usage[0].usage_metadata["output_token_details"]["reasoning"], 4081)
        self.assertEqual(runtime.execution.tool_call_count, 0)
        self.assertEqual([item["tool_choice"] for item in wire], ["auto", "auto"])
        self.assertTrue(all(item["thinking"] == {"type": "enabled", "clear_thinking": True} for item in wire))
        self.assertTrue(all(item["reasoning_effort"] == "low" for item in wire))
        self.assertTrue(all("clear_thinking" not in item for item in wire))
        self.assertTrue(all(item["tool_stream"] is True for item in wire))
        self.assertTrue(all(item["max_tokens"] == 4096 for item in wire))
        self.assertTrue(all("max_completion_tokens" not in item for item in wire))
        self.assertTrue(all("StageResult" in {tool["function"]["name"] for tool in item["tools"]} for item in wire))
        retries = [item.checkpoint["channel_values"].get("structured_output_retries")
                   for item in saver.list({"configurable": {"thread_id": "stage-run"}})]
        self.assertIn(1, retries, "The repair count must be checkpointed, not hidden in an in-memory retry loop")
        self.assertEqual(retries[0], 0)
        self.assertFalse(any(event.kind == "answer.delta" for event in events))


if __name__ == "__main__":
    unittest.main()
