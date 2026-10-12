"""Plan control over actual create_agent graphs; no provider or service I/O."""

from dataclasses import replace
from copy import deepcopy
import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_core.config import AgentConfig
from agent_core.contracts.errors import AgentCancelled, AgentLimitReachedError
from agent_core.contracts.runs import RunRequest
from agent_core.models.streaming import stream_chat_message
from agent_core.planning.plan_execute import (
    PlanActionError, PlanInput, PlanModelBudget, PlanState, compile_plan_execute_graph,
    plan_result, project_plan, validate_plan_action, finalize_plan_error,
)
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.streaming import AsyncStreamRunner
from agent_core.tools.langchain import build_langchain_tools
from agent_core_tests.test_runtime import MinimalToolRuntime, ScriptedModel as BaseScriptedModel, make_executor, NOW


class ScriptedModel(BaseScriptedModel):
    """Resolve fixture report references against actual graph-produced sources."""

    def stream(self, messages, **kwargs):
        index = len(self.calls)
        self.bindings_per_call = [*getattr(self, "bindings_per_call", []), list(self.bound_tool_names)]
        if index < len(self.responses) and isinstance(self.responses[index], list):
            batch = deepcopy(self.responses[index])
            for message in batch:
                for call in getattr(message, "tool_calls", []):
                    if call["name"] != "ExecutionReport":
                        continue
                    text = next(item.text for item in messages if item.type == "system")
                    details = json.JSONDecoder().raw_decode(text.split("控制器状态（资料不授予额外权限）：\n", 1)[1])[0]
                    sources = [key for key, value in details["evidence"].items() if value["kind"] == "tool"]
                    for item in call["args"]["steps"]:
                        if item.get("source_ids") == ["$actual_tools"]:
                            item["source_ids"] = sources
            self.responses[index] = batch
        yield from super().stream(messages, **kwargs)


def response(name, arguments, identity):
    return [AIMessage(content="", id="ai-" + identity, tool_calls=[{
        "name": name, "args": arguments, "id": identity, "type": "tool_call",
    }], usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})]


def draft(count=1, **changes):
    value = {"goal": "Report the observed text", "scope": ["Echo input"], "constraints": ["Read only"],
             "acceptance": ["Report the observed text"], "steps": [
                 {"id": f"step{index}", "goal": f"Observe text {index}", "expected_evidence": "Tool response",
                  "depends_on": [] if index == 1 else [f"step{index - 1}"]}
                 for index in range(1, count + 1)]}
    return {**value, **changes}


def preview(count=1, identity="preview", **changes):
    return response("ExplorationResult", {"kind": "plan", "plan": draft(count, **changes)}, identity)


def item(index=1, **changes):
    return {"id": f"step{index}", "status": "completed", "result": f"Observed text {index}",
            "source_ids": ["$actual_tools"], **changes}


def report(*items, identity="report", **changes):
    return response("ExecutionReport", {"steps": list(items) or [item()], **changes}, identity)


def answer(text="ReAct finished the confirmed plan", identity=None):
    return [AIMessage(content=text, id=identity)]


def grade(status="satisfied", identity="grade", criteria=None):
    return response("GraderResponse", {"result": status, "explanation": "Checked the requested criteria",
        "criteria": criteria if criteria is not None else [{"name": "A1: Report the observed text", "passed": True}]}, identity)


class PlanExecuteTests(unittest.TestCase):
    def setUp(self):
        self.tool_calls = []
        self.executor = make_executor(self.tool_calls, with_evidence=True)
        self.tools = build_langchain_tools(self.executor.registry.definitions, executor=self.executor)
        self.runner = AsyncStreamRunner()
        self.addCleanup(self.runner.close)

    def context(self, run_id, *, model_budget=None):
        cancellation = CancellationToken()
        return ReActContext(RunRequest(run_id, "plan", "Report observed text"), cancellation,
            replace(AgentConfig(), max_planning_rounds=8), self.runner, lambda: NOW,
            MinimalToolRuntime(cancellation), ReActPolicy(), model_budget=model_budget or PlanModelBudget())

    def graph(self, model, action, *, saver=None, **kwargs):
        def collect(*args, phase=None, step_id=None, **options):
            return stream_chat_message(*args, **options)
        return compile_plan_execute_graph(model, self.tools, action=action,
            stream_chat_message=collect, policy=ReActPolicy(), checkpointer=saver, **kwargs)

    def invoke(self, model, action=None, prior=None, run_id="run1", *, inherited=None, **kwargs):
        context = self.context(run_id)
        graph = self.graph(model, action or PlanInput(), **kwargs)
        state = {"references": [], "observation_bytes": 0, **(inherited or {}),
                 "messages": [HumanMessage(content="Report observed text", id="user-" + run_id)], "plan_state": prior}
        return self.runner.run(graph.ainvoke(state, context=context, config={"configurable": {"thread_id": run_id}, "recursion_limit": 150})), context

    def stream(self, graph, *args, **kwargs):
        events = graph.astream(*args, **kwargs)
        try:
            while True:
                try:
                    yield self.runner.run(anext(events))
                except StopAsyncIteration:
                    return
        finally:
            self.runner.run(events.aclose())

    def approved(self, plan):
        return PlanInput(kind="execute_plan", plan_id=plan["plan_id"], expected_revision=plan["revision"])

    def test_start_plan_opens_a_new_budget_window_and_keeps_only_task_context(self):
        from deepagents.backends.utils import create_file_data, file_data_to_string
        from agent_core.memory import TASK_NOTES_PATH, TASK_NOTES_TEMPLATE, inherit_task_memory
        from agent_core.runtime.budget import ExecutionBudget
        from agent_core_tests.test_runtime import GenericEvidence

        ready, _ = self.invoke(ScriptedModel([preview(2)]))
        prior = deepcopy(ready["plan_state"])
        prior.update(goal="Old goal", scope=["Old scope"], constraints=["Old constraint"], acceptance=["Old acceptance"])
        meter = PlanModelBudget()
        for _ in range(64):
            meter.begin_call()
            meter.record_usage({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12})
        totals = {"tool_call_count": 24, "observation_bytes": 98304, "calls_by_tool": {"echo_text": 4},
                  "calls_by_category": {"external": 8, "memory": 16}}
        old_references = [GenericEvidence(f"R{index}") for index in range(1, 10)]
        old_execution = {**ExecutionBudget().snapshot(), **totals, "revision": 200,
                         "references": old_references, "stop_reason": "tool_call_limit"}
        prior.update(model_ledger=meter.snapshot(), tool_totals=totals, phase="failed", approved_revision=prior["revision"],
                     approved_scope={key: prior[key] for key in ("goal", "scope", "constraints", "acceptance")},
                     sources={"old:proof": {"kind": "tool", "content": "Old result"}}, replans=3)
        prior["steps"][0].update(status="completed", result="Old result", source_ids=["old:proof"])
        old_copy = deepcopy(prior)
        notes = TASK_NOTES_TEMPLATE.replace("待记录。", "Prior task context", 1)
        inherited = inherit_task_memory({"plan_state": prior, "references": old_references,
            "files": {TASK_NOTES_PATH: create_file_data(notes)}, "todos": [{"content": "Old work", "status": "completed"}],
            "task_source_ids": ["old:proof"]})
        # Even an over-broad host inheritance cannot reactivate old accounting.
        inherited.update(model_ledger=meter.snapshot(), execution_ledger=old_execution,
                         observation_bytes=98304, tool_call_count=24)
        action = PlanInput(kind="start_plan", plan_id=prior["plan_id"], expected_revision=prior["revision"])
        new, context = self.invoke(ScriptedModel([
            response("echo_text", {"text": "fresh evidence"}, "fresh"), preview(identity="new-plan", goal="New goal")]),
            action, prior, "new-window", inherited=inherited, exploration_tools=self.tools)
        plan = new["plan_state"]
        self.assertNotEqual(plan["plan_id"], prior["plan_id"])
        self.assertEqual(plan["revision"], 1)
        self.assertEqual(plan["phase"], "ready")
        self.assertEqual(plan["goal"], "New goal")
        self.assertEqual(plan["scope"], ["Echo input"])
        self.assertEqual(plan["constraints"], ["Read only"])
        self.assertEqual(plan["acceptance"], ["Report the observed text"])
        self.assertIsNone(plan["approved_revision"])
        self.assertIsNone(plan["approved_scope"])
        self.assertEqual(plan["step_history"], [])
        self.assertEqual(plan["replans"], 0)
        self.assertTrue(all(step["status"] == "pending" and not step["result"] and not step["source_ids"] for step in plan["steps"]))
        self.assertNotIn("old:proof", plan["sources"])
        self.assertEqual(plan["budget"]["modelCalls"], 2)
        self.assertEqual(plan["budget"]["toolCalls"], 1)
        self.assertEqual(context.tool_runtime.execution.calls_by_category, {"external": 1})
        self.assertEqual(context.tool_runtime.execution.calls_by_tool, {"echo_text": 1})
        self.assertLess(new["observation_bytes"], 98304)
        self.assertEqual([reference.reference_id for reference in new["references"]], ["R1"])
        self.assertEqual(file_data_to_string(new["files"][TASK_NOTES_PATH]), notes)
        self.assertEqual(new["task_source_ids"], ["old:proof"])
        self.assertEqual(new["todos"], [])
        self.assertEqual(prior, old_copy)
        self.assertEqual(self.tool_calls, ["fresh evidence"])
        with self.assertRaises(PlanActionError):
            self.invoke(ScriptedModel([]), action, plan, "stale-start")
        with self.assertRaises(PlanActionError):
            self.invoke(ScriptedModel([]), self.approved(prior), plan, "stale-execute")
        with self.assertRaisesRegex(PlanActionError, "new Run"):
            self.invoke(ScriptedModel([]), PlanInput(kind="start_plan", plan_id=plan["plan_id"], expected_revision=1), plan, "new-window")
        revised, continued = self.invoke(ScriptedModel([preview(identity="supplement")]),
            PlanInput(kind="message", plan_id=plan["plan_id"], expected_revision=1), plan, "same-plan-supplement",
            inherited=inherit_task_memory(new))
        self.assertEqual(revised["plan_state"]["plan_id"], plan["plan_id"])
        self.assertEqual(revised["plan_state"]["revision"], 2)
        self.assertEqual(continued.model_budget.calls, 3)
        self.assertEqual(continued.tool_runtime.execution.tool_call_count, 1)

    def test_start_plan_without_prior_plan_and_same_run_resume_preserve_the_new_window(self):
        ready, _ = self.invoke(ScriptedModel([preview()]), PlanInput(kind="start_plan"), run_id="old-plan")
        self.assertEqual(ready["plan_state"]["revision"], 1)
        old = ready["plan_state"]
        action = PlanInput(kind="start_plan", plan_id=old["plan_id"], expected_revision=old["revision"])
        model = ScriptedModel([response("echo_text", {"text": "once"}, "once"),
                               RuntimeError("new-plan checkpoint boundary"), preview(identity="recovered")])
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "new-plan-resume"}, "recursion_limit": 150}
        first_context = self.context("new-plan-resume")
        graph = self.graph(model, action, saver=saver, exploration_tools=self.tools)
        with self.assertRaisesRegex(RuntimeError, "checkpoint boundary") as failure:
            self.runner.run(graph.ainvoke({"messages": [HumanMessage("Start a different plan", id="new-plan-user")], "plan_state": old},
                         context=first_context, config=config, durability="sync"))
        finalize_plan_error(graph, config, first_context, failure.exception)
        interrupted = graph.get_state(config).values["plan_state"]
        self.assertNotEqual(interrupted["plan_id"], old["plan_id"])
        self.assertEqual(interrupted["budget"]["toolCalls"], 1)
        self.assertEqual(interrupted["budget"]["modelCalls"], 2)
        resumed_context = self.context("new-plan-resume")
        rebuilt = self.graph(model, action, saver=saver, exploration_tools=self.tools)
        result = self.runner.run(rebuilt.ainvoke(None, context=resumed_context, config=config, durability="sync"))
        self.assertEqual(result["plan_state"]["plan_id"], interrupted["plan_id"])
        self.assertEqual(result["plan_state"]["revision"], 1)
        self.assertEqual(result["plan_state"]["phase"], "ready")
        self.assertEqual(resumed_context.model_budget.calls, 3)
        self.assertEqual(resumed_context.tool_runtime.execution.tool_call_count, 1)
        self.assertEqual(self.tool_calls, ["once"])

    def test_new_plan_preserves_original_history_source_run_as_context(self):
        context = self.context("new-history-run")
        graph = self.graph(ScriptedModel([preview()]), PlanInput(kind="start_plan"))
        result = self.runner.run(graph.ainvoke({"messages": [
            HumanMessage("Prior task context", id="old-history-user", additional_kwargs={
                "history_source_id": "old-history-source", "history_run_id": "old-history-run",
                "history_backfill": True}),
            HumanMessage("Plan the new goal", id="current-user"),
        ]}, context=context, config={"configurable": {"thread_id": "new-history-run"}}))
        plan = result["plan_state"]
        self.assertEqual(plan["sources"]["old-history-source"]["run_id"], "old-history-run")
        self.assertEqual(plan["sources"]["old-history-source"]["kind"], "user")
        self.assertEqual(plan["sources"]["new-history-run:current-user"]["run_id"], "new-history-run")
        self.assertTrue(all(not step["source_ids"] for step in plan["steps"]))
        self.assertEqual(plan["budget"]["toolCalls"], 0)
        self.assertEqual(result["references"], [])

    def test_context_preparation_runs_after_new_window_and_is_not_repeated_on_resume(self):
        from agent_core.context.summary import ContextSummarizer

        ready, _ = self.invoke(ScriptedModel([preview()]))
        prior = ready["plan_state"]
        action = PlanInput(kind="start_plan", plan_id=prior["plan_id"], expected_revision=prior["revision"])
        context = self.context("prepared-plan")
        model = ScriptedModel([[AIMessage("Earlier context, not current execution", usage_metadata={
            "input_tokens": 4, "output_tokens": 2, "total_tokens": 6})],
            RuntimeError("after prepared checkpoint"), preview(identity="resumed-plan")])

        def collect(*args, phase=None, step_id=None, **options):
            return stream_chat_message(*args, **options)

        summarizer = ContextSummarizer(model, collect, context)
        prepared_calls = []

        def prepare(state, active_context):
            prepared_calls.append(state["plan_state"]["plan_id"])
            self.assertEqual(active_context.model_budget.calls, 0)
            self.assertEqual(state["summary_calls"], 0)
            self.assertEqual(state["context_summary"], {})
            text = summarizer.summarize([HumanMessage("Archived history", id="archived-source")])
            return {"messages": [HumanMessage("New plan goal")], "summary_calls": summarizer.calls,
                    "context_summary": {"text": text, "source_ids": ["archived-source"],
                        "covered_message_ids": [], "has_legacy_dependencies": False}}

        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "prepared-plan"}, "recursion_limit": 150}
        graph = self.graph(model, action, saver=saver, context_summarizer=summarizer, prepare_context=prepare)
        with self.assertRaisesRegex(RuntimeError, "prepared checkpoint") as failure:
            self.runner.run(graph.ainvoke({"messages": [], "plan_state": prior, "summary_calls": 4,
                          "context_summary": {"text": "Old plan progress"}}, config=config, context=context))
        finalize_plan_error(graph, config, context, failure.exception)
        saved = graph.get_state(config).values
        self.assertEqual(saved["plan_state"]["budget"]["modelCalls"], 2)
        self.assertEqual(saved["summary_calls"], 1)
        self.assertEqual(saved["context_summary"]["text"], "Earlier context, not current execution")
        user = next(message for message in saved["messages"] if isinstance(message, HumanMessage))
        self.assertIsNotNone(user.id)
        self.assertEqual(user.text, "New plan goal")
        self.assertIn(f"prepared-plan:{user.id}", saved["plan_state"]["sources"])
        resumed_context = self.context("prepared-plan")
        rebuilt = self.graph(model, action, saver=saver, prepare_context=prepare,
            context_summarizer=ContextSummarizer(model, collect, resumed_context))
        result = self.runner.run(rebuilt.ainvoke(None, config=config, context=resumed_context))
        self.assertEqual(prepared_calls, [result["plan_state"]["plan_id"]])
        self.assertEqual(result["summary_calls"], 1)
        self.assertEqual(result["plan_state"]["budget"]["modelCalls"], 3)
        self.assertEqual(result["plan_state"]["phase"], "ready")

    def test_stage_context_metadata_returns_to_root_and_survives_failed_report(self):
        from langchain.agents.middleware import AgentMiddleware

        observations = []
        metadata = {"context_summary": {"text": "Earlier derived context", "source_ids": [],
                        "covered_message_ids": [], "has_legacy_dependencies": False},
                    "summary_calls": 2, "context_history_dependencies": ["memory:fixture:v1"],
                    "context_has_legacy_history": True, "run_exposed_memory_sources": ["memory:fixture:v1"],
                    "run_memory_exposure_complete": False}

        class ContextMetadata(AgentMiddleware):
            def before_model(self, state, runtime):
                if state.get("plan_stage") == "execute":
                    return metadata
                if state.get("plan_stage") == "report":
                    observations.append({key: state.get(key) for key in metadata})

        ready, _ = self.invoke(ScriptedModel([preview()]))
        model = ScriptedModel([response("echo_text", {"text": "observed"}, "observed"), answer(),
                               RuntimeError("report unavailable")])
        context = self.context("metadata-plan")
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "metadata-plan"}, "recursion_limit": 150}
        graph = self.graph(model, self.approved(ready["plan_state"]), saver=saver, middleware=[ContextMetadata()])
        with self.assertRaisesRegex(RuntimeError, "report unavailable") as failure:
            self.runner.run(graph.ainvoke({"messages": [HumanMessage("Execute", id="execute")], "plan_state": ready["plan_state"]},
                         context=context, config=config))
        finalize_plan_error(graph, config, context, failure.exception)
        saved = graph.get_state(config).values
        self.assertEqual(observations, [metadata])
        self.assertEqual({key: saved[key] for key in metadata}, metadata)
        self.assertTrue(any("Earlier derived context" in message.text for message in model.calls[-1][1]))

    def test_preview_requires_explicit_current_revision_then_real_tools_and_rubric(self):
        model = ScriptedModel([preview(), response("echo_text", {"text": "hello"}, "echo1"), answer(), report(),
                               [AIMessage(content="Observed text 1", id="final")], grade()])
        ready, first_context = self.invoke(model)
        self.assertEqual(ready["plan_state"]["phase"], "ready")
        self.assertEqual(self.tool_calls, [])
        self.assertEqual(first_context.model_budget.calls, 1)
        preview_result = plan_result(ready)
        self.assertEqual(preview_result.metadata["artifactKind"], "plan_preview")
        self.assertEqual(preview_result.output, ready["final_answer"])
        self.assertIn("**目标**：Report the observed text", preview_result.output)
        self.assertIn("`step1` Observe text 1；证据：Tool response", preview_result.output)
        self.assertIn("**验收标准**", preview_result.output)
        self.assertIn("待确认", preview_result.output)
        done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(done["plan_state"]["validation"]["status"], "satisfied")
        self.assertEqual(done["plan_state"]["validation"]["iterations"], 1)
        self.assertEqual(self.tool_calls, ["hello"])
        self.assertEqual(context.model_budget.calls, 6)
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 1)
        self.assertEqual(len(plan_result(done).artifacts), 1)
        self.assertTrue(done["plan_state"]["steps"][0]["source_ids"][0].startswith("run2:"))
        self.assertEqual(done["plan_state"]["budget"]["unknownUsageCalls"], 2)
        self.assertIsNone(done["plan_state"]["budget"]["totalTokens"])

    def test_whole_plan_is_one_react_loop_then_one_report_and_only_validated_answer_is_published(self):
        from langchain.agents.middleware import AgentMiddleware

        class PrivateStageOutput(AgentMiddleware):
            def before_model(self, state, runtime):
                runtime.stream_writer({"kind": "answer.delta", "payload": {"delta": "unapproved stage output"}})
                runtime.stream_writer({"kind": "task.updated", "payload": {"private": "unapproved stage output"}})
                runtime.stream_writer({"provider_trace": "unapproved stage output"})

        ready, _ = self.invoke(ScriptedModel([preview(2)]))
        model = ScriptedModel([
            response("echo_text", {"text": "one"}, "first"),
            response("echo_text", {"text": "two"}, "second"), answer("Private candidate before report"),
            report(item(1), item(2)), answer("Validated whole-plan result"), grade(),
        ])
        context = self.context("whole-run")
        graph = self.graph(model, self.approved(ready["plan_state"]), middleware=[PrivateStageOutput()])
        self.assertNotIn("select_step", graph.get_graph().nodes)
        self.assertNotIn("execute_step", graph.get_graph().nodes)
        events = list(self.stream(graph, {"messages": [HumanMessage("Execute", id="whole-user")],
                                   "plan_state": ready["plan_state"]}, context=context,
                                  config={"configurable": {"thread_id": "whole-run"}, "recursion_limit": 150},
                                  stream_mode=["values", "custom"]))
        done = [data for kind, data in events if kind == "values"][-1]
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(self.tool_calls, ["one", "two"])
        self.assertTrue(all("ExecutionReport" not in names and "StepResult" not in names
                            for names in model.bindings_per_call[:3]))
        self.assertEqual(model.bindings_per_call[3], ["ExecutionReport"])
        self.assertIn("Observe text 1", model.calls[0][1][0].text)
        self.assertIn("Observe text 2", model.calls[0][1][0].text)
        published = [data["payload"]["delta"] for kind, data in events
                     if kind == "custom" and data.get("kind") == "answer.delta"]
        self.assertEqual("".join(published), "Validated whole-plan result")
        self.assertEqual(context.model_budget.calls, 7)

        custom = [data for kind, data in events if kind == "custom"]
        self.assertNotIn("unapproved stage output", str(custom))
        phases = [data["payload"] for data in custom if data["kind"] == "step.state"
                  and data["payload"]["stepId"] == "context_validation" and data["payload"]["status"] == "running"]
        self.assertEqual([phase["activityLabel"] for phase in phases], ["计划执行", "执行结果核对", "最终验收"])
        self.assertEqual(len({phase["activityScopeId"] for phase in phases}), 3)
        tool_events = [data for data in custom if data["kind"] in {"tool.planned", "tool.started", "tool.completed"}]
        self.assertEqual(len(tool_events), 6)
        self.assertTrue(all(event["activityScopeId"] == phases[0]["activityScopeId"] for event in tool_events))
        self.assertTrue(all(event["activityLabel"] == "计划执行" for event in tool_events))
        step_events = [data for data in custom if data["kind"] == "plan.step"]
        self.assertEqual([event["payload"]["status"] for event in step_events], ["completed", "completed"])
        self.assertTrue(all(event["activityLabel"] == "计划步骤" for event in step_events))

    def test_activity_scope_survives_same_run_resume_and_changes_for_replanning(self):
        ready, _ = self.invoke(ScriptedModel([preview()]))
        model = ScriptedModel([response("echo_text", {"text": "observed"}, "proof"),
            RuntimeError("interrupted model"), answer(),
            report(item(status="blocked", unresolved=["Need a final check"])),
            response("ReplanDecision", {"kind": "continue", "reason": "Check the same evidence", "remaining_steps": draft()["steps"]}, "replan"),
            answer(), report(), answer("Validated result"), grade()])
        action = self.approved(ready["plan_state"])
        saver = InMemorySaver()
        config = {"configurable": {"thread_id": "activity-resume"}, "recursion_limit": 150}
        context = self.context("activity-resume")
        graph = self.graph(model, action, saver=saver)
        before = []
        with self.assertRaisesRegex(RuntimeError, "interrupted model") as failure:
            for kind, data in self.stream(graph, {"messages": [HumanMessage("Execute", id="activity-user")],
                "plan_state": ready["plan_state"]}, config=config, context=context, stream_mode=["values", "custom"]):
                if kind == "custom":
                    before.append(data)
        finalize_plan_error(graph, config, context, failure.exception)
        resumed = self.context("activity-resume")
        rebuilt = self.graph(model, action, saver=saver)
        after = list(self.stream(rebuilt, None, config=config, context=resumed, stream_mode=["values", "custom"]))
        done = [data for kind, data in after if kind == "values"][-1]
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(self.tool_calls, ["observed"])
        self.assertEqual(resumed.tool_runtime.execution.tool_call_count, 1)
        original = next(event["payload"] for event in before if event["kind"] == "step.state"
                        and event["payload"]["stepId"] == "agent_decision_2")
        steps = [data["payload"] for kind, data in after if kind == "custom" and data["kind"] == "step.state"]
        recovered = next(step for step in steps if step["stepId"] == "agent_decision_2" and step["status"] == "running")
        self.assertEqual(recovered["activityScopeId"], original["activityScopeId"])
        next_execute = next(step for step in steps if step["stepId"] == "context_validation"
                            and step["activityLabel"] == "计划执行")
        self.assertNotEqual(next_execute["activityScopeId"], original["activityScopeId"])
        self.assertNotIn("checkpoint", original["activityScopeId"])

    def test_failed_approved_execution_preserves_sources_and_continues_in_a_new_run(self):
        from agent_core.contracts.errors import AgentUpstreamError
        from agent_core.memory import inherit_task_memory
        ready, _ = self.invoke(ScriptedModel([preview()]))
        failed, first_context = self.invoke(ScriptedModel([
            response("echo_text", {"text": "already observed"}, "observed"), AgentUpstreamError("provider failed"),
        ]), self.approved(ready["plan_state"]), ready["plan_state"], "failure-run")
        plan = failed["plan_state"]
        self.assertEqual(plan["phase"], "failed")
        self.assertEqual(plan["steps"][0]["status"], "blocked")
        self.assertEqual(plan["step_history"][0]["status"], "blocked")
        self.assertTrue(any(value["kind"] == "tool" for value in plan["sources"].values()))
        self.assertEqual(project_plan(plan)["approvedRevision"], plan["revision"])
        self.assertEqual(first_context.model_budget.calls, 3)
        model = ScriptedModel([answer("Reuse the saved observation"), report(), answer("Observed existing data"), grade()])
        done, context = self.invoke(model, self.approved(plan), plan, "continue-run", inherited=inherit_task_memory(failed))
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(self.tool_calls, ["already observed"])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 1)
        self.assertEqual(context.model_budget.calls, 7)
        unapproved = {**plan, "approved_revision": None, "approved_scope": None}
        with self.assertRaises(PlanActionError):
            validate_plan_action(unapproved, self.approved(unapproved))

    def test_report_cannot_omit_steps_or_claim_completion_without_sources(self):
        for reported in (report(item(1)), report(item(1, source_ids=[]), item(2))):
            with self.subTest(report=reported[0].tool_calls[0]["args"]):
                ready, _ = self.invoke(ScriptedModel([preview(2)]))
                responses = [response("echo_text", {"text": "observed"}, "observation"), answer(), reported]
                if len(reported[0].tool_calls[0]["args"]["steps"]) == 2:
                    responses.append(response("ReplanDecision", {"kind": "stop", "reason": "Evidence is incomplete"}, "stop"))
                else:
                    responses += [report(item(1), identity="missing-again"), report(item(1), identity="missing-final")]
                failed, context = self.invoke(ScriptedModel(responses), self.approved(ready["plan_state"]), ready["plan_state"], "report-run")
                self.assertNotEqual(failed["plan_state"]["phase"], "completed")
                self.assertTrue(all(step["status"] != "completed" for step in failed["plan_state"]["steps"]))
                self.assertEqual(failed["plan_state"]["validation"]["status"], "not_run")
                if len(reported[0].tool_calls[0]["args"]["steps"]) == 1:
                    self.assertEqual(context.model_budget.calls, 6)  # preview, execution, then at most three reports

    def test_report_missing_steps_repairs_in_place_without_repeating_execution(self):
        ready, _ = self.invoke(ScriptedModel([preview(3)]))
        model = ScriptedModel([
            response("echo_text", {"text": "observed"}, "observation"), answer("Only a summary was obtained"),
            report(item(3)),
            report(item(1, status="blocked", result="No full list", unresolved=["Missing pages"]),
                   item(2, status="blocked", result="No per-item evidence", unresolved=["Missing details"]),
                   item(3, result="Summary only", unresolved=["Summary does not prove full coverage"]), identity="corrected"),
            response("ReplanDecision", {"kind": "stop", "reason": "Required evidence is missing"}, "stop"),
        ])
        failed, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "repair-run")
        self.assertEqual(failed["plan_state"]["phase"], "partial")
        self.assertEqual([step["status"] for step in failed["plan_state"]["steps"]], ["blocked"] * 3)
        self.assertEqual(failed["plan_state"]["validation"]["status"], "not_run")
        self.assertEqual(self.tool_calls, ["observed"])
        self.assertEqual(context.model_budget.calls, 6)
        self.assertEqual(model.bindings_per_call[3], ["ExecutionReport"])
        feedback = next(message for message in model.calls[3][1]
                        if message.type == "tool" and "execution_report_step_mismatch" in message.text)
        payload = json.loads(feedback.text)
        self.assertEqual(payload["required_step_ids"], ["step1", "step2", "step3"])
        self.assertEqual(payload["missing_step_ids"], ["step1", "step2"])
        self.assertEqual(payload["unexpected_step_ids"], [])
        for key in ("goal", "scope", "constraints", "acceptance"):
            self.assertEqual(failed["plan_state"][key], ready["plan_state"][key])

    def test_report_cannot_rewrite_completed_steps_and_repair_can_complete_remaining_work(self):
        ready, _ = self.invoke(ScriptedModel([preview(2)]))
        prior = deepcopy(ready["plan_state"])
        prior["sources"] = {"prior:proof": {"kind": "tool", "tool_name": "echo_text", "content": "First observation"}}
        prior["steps"][0].update(status="completed", result="First observation", source_ids=["prior:proof"])
        completed = deepcopy(prior["steps"][0])
        model = ScriptedModel([
            response("echo_text", {"text": "second"}, "second"), answer(),
            report(item(1, result="Attempted overwrite"), item(2)),
            report(item(2), identity="corrected"), answer("All observations verified"), grade(),
        ])
        done, context = self.invoke(model, self.approved(prior), prior, "preserve-run")
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(done["plan_state"]["steps"][0], completed)
        self.assertEqual(self.tool_calls, ["second"])
        self.assertEqual(context.model_budget.calls, 7)
        feedback = next(message for message in model.calls[3][1]
                        if message.type == "tool" and "execution_report_step_mismatch" in message.text)
        payload = json.loads(feedback.text)
        self.assertEqual(payload["required_step_ids"], ["step2"])
        self.assertEqual(payload["unexpected_step_ids"], ["step1"])
        rejected = next(message for message in model.calls[3][1] if message.id == "ai-report")
        self.assertEqual(rejected.usage_metadata, {"input_tokens": 4, "output_tokens": 2, "total_tokens": 6})
        self.assertEqual(context.model_budget.input_tokens, 20)
        self.assertEqual(context.model_budget.output_tokens, 10)

    def test_report_repair_rejects_hallucinated_native_reads_before_io(self):
        from unittest.mock import patch
        from deepagents.backends import StateBackend
        from deepagents.backends.utils import create_file_data
        from langgraph.store.memory import InMemoryStore
        from agent_core.contracts.errors import ToolPolicyDeniedError
        from agent_core.memory import MemoryBinding, MemoryScope
        from agent_core_tests.test_memory_components import SafeHistory, ScopedService

        class History(SafeHistory):
            def __init__(self):
                self.read_calls = []

            def read(self, *args, **kwargs):
                self.read_calls.append((args, kwargs))
                return []

        original_read = StateBackend.read
        for name, arguments in (("read_file", {"file_path": "/scratch/forbidden", "limit": 1}),
                                ("history_read", {"source_ids": ["prior:proof"], "max_characters": 100})):
            with self.subTest(tool=name):
                store = InMemoryStore()
                history = History()
                scope = MemoryScope("tenant", "user", None, "session", "task", "report-no-io")
                binding = MemoryBinding(scope, store, ScopedService(store), history=history)
                ready, _ = self.invoke(ScriptedModel([preview(2)]))
                prior = {**ready["plan_state"], "sources": {
                    "prior:proof": {"kind": "tool", "tool_name": "echo_text", "content": "Saved observation"}}}
                model = ScriptedModel([answer("Existing evidence only"), report(item(2)),
                                       response(name, arguments, "forbidden-read")])
                context = replace(self.context("report-no-io"), memory=binding)
                graph = self.graph(model, self.approved(prior), memory=binding, store=store)
                file_reads = []

                def record_read(backend, path, *args, **kwargs):
                    file_reads.append(path)
                    return original_read(backend, path, *args, **kwargs)

                with patch.object(StateBackend, "read", autospec=True, side_effect=record_read):
                    with self.assertRaisesRegex(ToolPolicyDeniedError, "report correction"):
                        self.runner.run(graph.ainvoke({"messages": [HumanMessage("Execute", id="report-no-io-user")], "plan_state": prior,
                                      "files": {"/scratch/forbidden": create_file_data("Must not read")}}, context=context,
                                     config={"configurable": {"thread_id": "report-no-io"}, "recursion_limit": 150}))
                self.assertEqual(model.bindings_per_call[2], ["ExecutionReport"])
                self.assertNotIn("/scratch/forbidden", file_reads)
                self.assertEqual(history.read_calls, [])
                self.assertEqual(context.model_budget.calls, 4)
                self.assertEqual(context.tool_runtime.execution.tool_call_count, 1)  # rejected attempt stays charged

    def test_report_repair_respects_remaining_plan_model_allowance(self):
        ready, _ = self.invoke(ScriptedModel([preview(2)]))
        model = ScriptedModel([response("echo_text", {"text": "observed"}, "observed"), answer(), report(item(2))])
        failed, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "limited-report", max_model_calls=4)
        self.assertEqual(failed["plan_state"]["phase"], "failed")
        self.assertEqual(context.model_budget.calls, 4)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(self.tool_calls, ["observed"])
        self.assertEqual(failed["plan_state"]["budget"]["toolCalls"], 1)
        self.assertEqual(failed["plan_state"]["validation"]["status"], "not_run")

    def test_unlimited_plan_budget_restores_35_calls_and_keeps_metering_past_64(self):
        previous = PlanModelBudget()
        usage = {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
        for _ in range(35):
            previous.begin_call()
            previous.record_usage(usage)
        continued = PlanModelBudget(max_calls=None)
        continued.restore(previous.snapshot())
        for _ in range(31):
            continued.begin_call()
            continued.record_usage(usage)
        self.assertEqual(continued.public(), {"modelCalls": 66, "maxModelCalls": None,
                         "inputTokens": 132, "outputTokens": 66, "totalTokens": 198, "unknownUsageCalls": 0})
        resumed = PlanModelBudget(max_calls=None)
        resumed.restore(continued.snapshot())
        resumed.begin_call()
        resumed.record_usage(None)
        self.assertEqual(resumed.public()["modelCalls"], 67)
        self.assertIsNone(resumed.public()["totalTokens"])
        self.assertEqual(resumed.public()["unknownUsageCalls"], 1)

    def test_workflow_none_keeps_unlimited_meter_and_finite_workflow_still_restricts_it(self):
        ready, _ = self.invoke(ScriptedModel([preview()]))
        prior = deepcopy(ready["plan_state"])
        previous = PlanModelBudget()
        for _ in range(64):
            previous.begin_call()
            previous.record_usage({"input_tokens": 2, "output_tokens": 1, "total_tokens": 3})
        prior["model_ledger"] = previous.snapshot()
        action = PlanInput(kind="message", plan_id=prior["plan_id"], expected_revision=prior["revision"])
        for maximum, expected_phase, expected_calls in ((None, "ready", 65), (64, "failed", 64)):
            with self.subTest(maximum=maximum):
                model = ScriptedModel([preview(identity="after-64")])
                context = self.context("unlimited-workflow", model_budget=PlanModelBudget(max_calls=None))
                graph = self.graph(model, action, max_model_calls=maximum)
                result = self.runner.run(graph.ainvoke({"messages": [HumanMessage("Refine the plan", id="refine-user")], "plan_state": prior},
                                      context=context, config={"configurable": {"thread_id": "unlimited-workflow"}, "recursion_limit": 150}))
                self.assertEqual(result["plan_state"]["phase"], expected_phase)
                self.assertEqual(result["plan_state"]["budget"]["modelCalls"], expected_calls)
                self.assertEqual(result["plan_state"]["budget"]["maxModelCalls"], maximum)
                self.assertEqual(len(model.calls), expected_calls - 64)
                self.assertEqual(result["plan_state"]["budget"]["maxReplans"], 3)

    def test_report_repair_counter_survives_same_run_checkpoint_recovery(self):
        ready, _ = self.invoke(ScriptedModel([preview(2)]))
        prior = ready["plan_state"]
        model = ScriptedModel([
            response("echo_text", {"text": "once"}, "once"), answer(), report(item(2), identity="bad1"),
            RuntimeError("simulated report boundary"), report(item(2), identity="bad2"), report(item(2), identity="bad3"),
        ])
        saver = InMemorySaver()
        action = self.approved(prior)
        config = {"configurable": {"thread_id": "report-resume"}, "recursion_limit": 150}
        first_context = self.context("report-resume")
        graph = self.graph(model, action, saver=saver)
        with self.assertRaisesRegex(RuntimeError, "simulated report boundary") as failure:
            self.runner.run(graph.ainvoke({"messages": [HumanMessage("Execute", id="report-resume-user")], "plan_state": prior},
                         context=first_context, config=config, durability="sync"))
        finalize_plan_error(graph, config, first_context, failure.exception)
        next_context = self.context("report-resume")
        rebuilt = self.graph(model, action, saver=saver)
        failed = self.runner.run(rebuilt.ainvoke(None, context=next_context, config=config, durability="sync"))
        self.assertEqual(failed["plan_state"]["phase"], "failed")
        self.assertEqual(self.tool_calls, ["once"])
        self.assertEqual(len(model.calls), 6)
        self.assertEqual(next_context.model_budget.calls, 7)  # preview plus six actual provider attempts
        self.assertEqual(failed["plan_state"]["validation"]["status"], "not_run")

    def test_plan_document_migration_is_explicit_and_keeps_approval_evidence_and_budget(self):
        from agent_core.planning.plan_execute import PLAN_STATE_VERSION, migrate_plan_state
        from agent_core.planning.plan_execute.contracts import parse_plan
        ready, _ = self.invoke(ScriptedModel([preview()]))
        old = {**ready["plan_state"], "version": "plan-execute-1", "active_step_id": "step1", "phase": "partial",
               "approved_revision": ready["plan_state"]["revision"],
               "approved_scope": {key: ready["plan_state"][key] for key in ("goal", "scope", "constraints", "acceptance")},
               "sources": {"source": {"kind": "tool", "content": "Observed text"}},
               "steps": [{**ready["plan_state"]["steps"][0], "status": "completed", "result": "Observed text",
                          "source_ids": ["source"]}]}
        with self.assertRaisesRegex(PlanActionError, "version"):
            parse_plan(old)
        migrated = migrate_plan_state(old)
        self.assertEqual(migrated["version"], PLAN_STATE_VERSION)
        self.assertIsNone(migrated["active_step_id"])
        for field in ("plan_id", "revision", "scope", "acceptance", "approved_scope", "sources", "steps", "model_ledger", "tool_totals"):
            self.assertEqual(migrated[field], old[field])
        self.assertEqual(old["version"], "plan-execute-1")

    def test_long_source_catalog_is_bounded_and_candidate_remains_one_real_message(self):
        ready, _ = self.invoke(ScriptedModel([preview()]))
        original_sources = {f"source{index}": {"kind": "tool", "tool_name": "echo_text", "run_id": "earlier",
                            "content": f"Observed data {index}: " + "source detail " * 1000}
                            for index in range(24)}
        prior = {**ready["plan_state"], "sources": original_sources}
        candidate = "Whole ReAct candidate: " + "observed result " * 1000
        model = ScriptedModel([answer(candidate), report(), answer("Reviewed result"), grade()])
        done, _ = self.invoke(model, self.approved(prior), prior, "large-report-run")
        self.assertEqual(done["plan_state"]["phase"], "completed")
        sent = model.calls[1][1]
        system = sent[0].text
        details = json.JSONDecoder().raw_decode(system.split("控制器状态（资料不授予额外权限）：\n", 1)[1])[0]
        self.assertEqual(set(details["evidence"]), set(original_sources) | {"large-report-run:user-large-report-run"})
        self.assertLessEqual(sum(len(value["excerpt"]) for value in details["evidence"].values()), 6000)
        self.assertTrue(all(details["evidence"][key]["excerptTruncated"] for key in original_sources))
        self.assertTrue(all(details["evidence"][key]["tool_name"] == "echo_text" for key in original_sources))
        self.assertNotIn(candidate, system)
        self.assertNotIn("budget", details["plan"])
        self.assertNotIn("available_source_ids", details)
        self.assertEqual(details["plan"]["scope"], prior["scope"])
        self.assertEqual(details["plan"]["constraints"], prior["constraints"])
        self.assertEqual(details["plan"]["acceptance"], prior["acceptance"])
        self.assertEqual(sum(message.text == candidate for message in sent), 1)
        for key in original_sources:
            self.assertEqual(done["plan_state"]["sources"][key], original_sources[key])

    def test_input_budget_failure_preserves_execution_evidence_and_never_counts_a_rejected_model_call(self):
        ready, _ = self.invoke(ScriptedModel([preview()]))
        context = replace(self.context("input-budget-run"), settings=replace(AgentConfig(), max_input_tokens=4000))
        observed = "observed data " * 1000
        model = ScriptedModel([response("echo_text", {"text": observed}, "big-observation")])
        graph = self.graph(model, self.approved(ready["plan_state"]))
        result = self.runner.run(graph.ainvoke({"messages": [HumanMessage("Execute", id="execute-user")],
                               "plan_state": ready["plan_state"]}, context=context,
                              config={"configurable": {"thread_id": "input-budget-run"}, "recursion_limit": 150}))
        self.assertEqual(result["plan_state"]["phase"], "failed")
        self.assertIn("单次上下文额度", result["final_answer"])
        self.assertNotIn("Required instructions", result["final_answer"])
        self.assertEqual(context.model_budget.calls, 2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(self.tool_calls, [observed])
        self.assertTrue(any(source["kind"] == "tool" for source in result["plan_state"]["sources"].values()))
        self.assertEqual(result["plan_state"]["budget"]["toolCalls"], 1)
        self.assertEqual(result["plan_state"]["validation"]["status"], "not_run")

    def test_repeated_truncated_stage_returns_safe_failure_and_keeps_consumption(self):
        truncated = [AIMessage(content="", response_metadata={"finish_reason": "length"},
            usage_metadata={"input_tokens": 12384, "output_tokens": 4096, "total_tokens": 16480,
                            "output_token_details": {"reasoning": 4081}})]
        result, context = self.invoke(ScriptedModel([truncated, truncated, truncated]))
        plan = result["plan_state"]
        self.assertEqual(plan["phase"], "failed")
        self.assertIn("输出达到长度上限", plan["unresolved"][0])
        self.assertEqual(plan["budget"]["modelCalls"], 3)
        self.assertEqual(plan["budget"]["totalTokens"], 49440)
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 0)
        self.assertEqual(plan["steps"], [])
        self.assertEqual(plan["validation"]["status"], "not_run")
        self.assertNotIn("reasoning", result["final_answer"])

    def test_question_answer_is_new_run_and_stale_revision_is_rejected_before_model(self):
        question = {"id": "target", "title": "Which text?", "options": [{"label": "hello", "description": "Echo hello"}]}
        model = ScriptedModel([response("ExplorationResult", {"kind": "questions", "questions": [question]}, "questions"), preview()])
        waiting, _ = self.invoke(model)
        plan = waiting["plan_state"]
        question_result = plan_result(waiting)
        self.assertEqual(question_result.metadata["artifactKind"], "questions")
        self.assertEqual(question_result.output, waiting["final_answer"])
        self.assertIn("Which text?", question_result.output)
        self.assertIn("hello：Echo hello", question_result.output)
        self.assertIn("不会自动执行", question_result.output)
        with self.assertRaises(PlanActionError):
            self.invoke(model, PlanInput(kind="answer_questions", plan_id=plan["plan_id"], expected_revision=plan["revision"] + 1,
                                        answers={"target": "hello"}), plan, "run2")
        self.assertEqual(len(model.calls), 1)
        action = PlanInput(kind="answer_questions", plan_id=plan["plan_id"], expected_revision=plan["revision"], answers={"target": "hello"})
        ready, context = self.invoke(model, action, plan, "run3")
        self.assertEqual(ready["plan_state"]["phase"], "ready")
        self.assertEqual(context.model_budget.calls, 2)
        self.assertEqual(self.tool_calls, [])
        self.assertIn('"target": "hello"', model.calls[1][1][0].text)

    def test_rubric_missing_acceptance_coverage_never_completes(self):
        for criterion in ("Other", "A1: A completely different criterion"):
            with self.subTest(criterion=criterion):
                model = ScriptedModel([preview(), response("echo_text", {"text": "hello"}, "echo1"), answer(), report(),
                    [AIMessage(content="Observed text 1")], grade(criteria=[{"name": criterion, "passed": True}])])
                ready, _ = self.invoke(model)
                done, _ = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
                self.assertEqual(done["plan_state"]["phase"], "partial")
                self.assertEqual(done["plan_state"]["validation"]["status"], "needs_revision")

    def test_budget_unknown_and_restore_do_not_turn_missing_usage_into_zero(self):
        budget = PlanModelBudget(max_calls=2)
        budget.begin_call()
        budget.record_usage(AIMessage(content="x", usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}))
        self.assertEqual(budget.public()["totalTokens"], 0)
        budget.begin_call()
        budget.record_usage(None)
        restored = PlanModelBudget(max_calls=2)
        restored.restore(budget.snapshot())
        self.assertEqual(restored.public()["unknownUsageCalls"], 1)
        self.assertIsNone(restored.public()["totalTokens"])
        with self.assertRaises(AgentLimitReachedError):
            restored.begin_call()

    def test_same_run_saver_resume_preserves_whole_react_tool_progress(self):
        model = ScriptedModel([preview(2), response("echo_text", {"text": "one"}, "echo1"),
            response("echo_text", {"text": "two"}, "echo2"), RuntimeError("simulated process boundary"), answer(), report(item(1), item(2)),
            [AIMessage(content="Observed both texts")], grade()])
        ready, _ = self.invoke(model)
        saver = InMemorySaver()
        action = self.approved(ready["plan_state"])
        graph = self.graph(model, action, saver=saver)
        config = {"configurable": {"thread_id": "resume-run"}, "recursion_limit": 150}
        initial = {"messages": [HumanMessage(content="Execute", id="execute-user")],
                   "plan_state": ready["plan_state"], "references": [], "observation_bytes": 0}
        first_context = self.context("resume-run")
        with self.assertRaisesRegex(RuntimeError, "simulated process") as error:
            self.runner.run(graph.ainvoke(initial, context=first_context, config=config, durability="sync"))
        finalize_plan_error(graph, config, first_context, error.exception)
        checkpoint = graph.get_state(config)
        self.assertEqual(checkpoint.values["plan_state"]["steps"][0]["status"], "blocked")
        self.assertTrue(checkpoint.values["plan_state"]["sources"])
        self.assertEqual(self.tool_calls, ["one", "two"])
        new_context = self.context("resume-run")
        rebuilt = self.graph(model, action, saver=saver)
        result = self.runner.run(rebuilt.ainvoke(None, context=new_context, config=config, durability="sync"))
        self.assertEqual(result["plan_state"]["phase"], "completed")
        self.assertEqual(self.tool_calls, ["one", "two"])
        self.assertEqual(new_context.tool_runtime.execution.tool_call_count, 2)
        self.assertEqual(len(result["references"]), 2)
        self.assertEqual(new_context.model_budget.calls, 8)

    def test_completed_step_and_evidence_numbering_survive_question_turns(self):
        from agent_core.memory import inherit_task_memory
        question = {"id": "detail", "title": "Confirm the second text", "options": []}
        model = ScriptedModel([preview(2), response("echo_text", {"text": "one"}, "echo1"),
            answer("Observed one; need clarification for two"),
            report(item(1), item(2, status="blocked", result="Second text unknown", unresolved=["Need second text"]),
                   needs_replan=True, new_conditions=["Need second text"]),
            response("ReplanDecision", {"kind": "questions", "reason": "Need user detail", "questions": [question]}, "ask-again"),
            preview(2, identity="updated-plan"), response("echo_text", {"text": "two"}, "echo2"), answer(), report(item(2)),
            [AIMessage(content="Both texts [R1] [R2]")], grade()])
        ready, _ = self.invoke(model)
        waiting, _ = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(waiting["plan_state"]["steps"][0]["status"], "completed")
        first_sources = waiting["plan_state"]["steps"][0]["source_ids"]
        action = PlanInput(kind="answer_questions", plan_id=waiting["plan_state"]["plan_id"],
                           expected_revision=waiting["plan_state"]["revision"], answers={"detail": "two"})
        updated, _ = self.invoke(model, action, waiting["plan_state"], "run3", inherited=inherit_task_memory(waiting))
        self.assertEqual(self.tool_calls, ["one"])
        result, context = self.invoke(model, self.approved(updated["plan_state"]), updated["plan_state"], "run4",
                                      inherited=inherit_task_memory(updated))
        self.assertEqual(result["plan_state"]["phase"], "completed")
        self.assertEqual(result["plan_state"]["steps"][0]["source_ids"], first_sources)
        self.assertEqual(self.tool_calls, ["one", "two"])
        self.assertEqual([reference.reference_id for reference in result["references"]], ["R1", "R2"])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 2)
        self.assertEqual(context.tool_runtime.execution.used_call_ids, {"echo2"})

    def test_official_notes_and_todo_root_checkpoint_survive_inner_failure(self):
        from deepagents.backends.utils import file_data_to_string
        from langgraph.store.memory import InMemoryStore
        from agent_core.memory import (MemoryBinding, MemoryScope, TASK_NOTES_PATH, TASK_NOTES_TEMPLATE,
                                       read_task_checkpoint, restore_memory_binding)
        from agent_core_tests.test_memory_components import ScopedService
        store = InMemoryStore()
        services = ScopedService(store)
        scope = MemoryScope("tenant", "user", None, "session", "task", "memory-run")
        binding = MemoryBinding(scope, store, services)
        # Use the official notes schema verbatim; an added source-free work item
        # does not pretend that external work has been verified.
        ready, _ = self.invoke(ScriptedModel([preview()]))
        old_text = TASK_NOTES_TEMPLATE.splitlines()[0]
        model = ScriptedModel([
            response("write_todos", {"todos": [{"content": "Observe text", "status": "in_progress"}]}, "todo"),
            response("edit_file", {"file_path": TASK_NOTES_PATH, "old_string": old_text, "new_string": old_text + " "}, "notes-edit"),
            RuntimeError("after official notes write"), answer(),
            response("read_file", {"file_path": TASK_NOTES_PATH}, "report-read"),
            report(), [AIMessage(content="Notes maintained")], grade()])
        context = replace(self.context("memory-run"), memory=binding)
        saver = InMemorySaver()
        graph = self.graph(model, self.approved(ready["plan_state"]), saver=saver, memory=binding, store=store)
        config = {"configurable": {"thread_id": "memory-run"}, "recursion_limit": 150}
        initial = {"messages": [HumanMessage(content="Execute", id="memory-user")], "plan_state": ready["plan_state"]}
        with self.assertRaisesRegex(RuntimeError, "after official") as error:
            self.runner.run(graph.ainvoke(initial, context=context, config=config, durability="sync"))
        finalize_plan_error(graph, config, context, error.exception)
        snapshot = graph.get_state(config).values
        self.assertEqual(snapshot["todos"][0]["status"], "in_progress")
        self.assertIn(TASK_NOTES_PATH, snapshot["files"])
        self.assertEqual(file_data_to_string(snapshot["files"][TASK_NOTES_PATH]).splitlines()[0], old_text + " ")
        task_view = read_task_checkpoint(saver, {"thread_id": "memory-run"})
        self.assertEqual(task_view["files"], snapshot["files"])
        new_binding = MemoryBinding(scope, store, services)
        self.assertTrue(restore_memory_binding(new_binding, snapshot))
        rebuilt = self.graph(model, self.approved(ready["plan_state"]), saver=saver, memory=new_binding, store=store)
        new_context = replace(self.context("memory-run"), memory=new_binding)
        done = self.runner.run(rebuilt.ainvoke(None, context=new_context, config=config, durability="sync"))
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(new_context.tool_runtime.execution.tool_call_count, 3)
        self.assertEqual(done["todos"][0]["status"], "in_progress")
        self.assertIn("read_file", model.bindings_per_call[4])
        self.assertFalse({"echo_text", "write_file", "edit_file", "write_todos"} & set(model.bindings_per_call[4]))
        self.assertTrue(any(message.type == "tool" and message.name == "read_file" and old_text in message.text
                            for message in model.calls[5][1]))

    def test_public_projection_excludes_private_scope_state(self):
        state = PlanState(plan_id="p", sources={"secret": {"content": "private source"}},
                          approved_scope={"secret": "private"}, model_ledger={"secret": "private"},
                          budget={"modelCalls": 3, "secret": "private"})
        public = project_plan(state)
        self.assertNotIn("private", str(public))
        self.assertEqual(public["budget"], {"modelCalls": 3})

    def test_cancellation_finalizer_persists_phase_and_consumed_model_budget(self):
        ready, _ = self.invoke(ScriptedModel([preview()]))
        context = self.context("cancel-run")

        class CancelModel(ScriptedModel):
            def stream(self, messages, **kwargs):
                self.calls.append(("model", list(messages)))
                context.cancellation.cancel()
                context.cancellation.raise_if_cancelled()
                yield

        model = CancelModel([])
        saver = InMemorySaver()
        graph = self.graph(model, self.approved(ready["plan_state"]), saver=saver)
        config = {"configurable": {"thread_id": "cancel-run"}, "recursion_limit": 150}
        with self.assertRaises(AgentCancelled) as error:
            self.runner.run(graph.ainvoke({"messages": [HumanMessage(content="Execute", id="cancel-user")], "plan_state": ready["plan_state"]},
                         context=context, config=config, durability="sync"))
        finalize_plan_error(graph, config, context, error.exception)
        state = graph.get_state(config).values
        self.assertEqual(state["plan_state"]["phase"], "cancelled")
        self.assertEqual(state["plan_state"]["budget"]["modelCalls"], 2)
        self.assertEqual(state["plan_state"]["budget"]["unknownUsageCalls"], 1)
        self.assertEqual(self.tool_calls, [])

    def test_pending_sibling_keeps_write_and_marks_uncertain_call_before_new_retry(self):
        from threading import Event
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import ToolMessage
        from langgraph.store.memory import InMemoryStore
        from agent_core.memory import MemoryBinding, MemoryScope, restore_memory_binding
        from agent_core_tests.test_memory_components import ScopedService
        store = InMemoryStore()
        services = ScopedService(store)
        scope = MemoryScope("tenant", "user", None, "session", "task", "pending-run")
        binding = MemoryBinding(scope, store, services)
        writes = []
        sibling_saved = Event()

        class PendingWriteSaver(InMemorySaver):
            def put_writes(self, config, pending, task_id, task_path=""):
                super().put_writes(config, pending, task_id, task_path)
                for channel, value in pending:
                    if channel == "messages" and isinstance(value, list) and any(
                        isinstance(message, ToolMessage) and message.tool_call_id == "written"
                        for message in value
                    ):
                        sibling_saved.set()

        class CrashRead(AgentMiddleware):
            async def awrap_tool_call(self, request, handler):
                if request.tool_call["name"] == "read_file":
                    # The gate serializes tool I/O; the official Send result
                    # commits afterward. Inject this fault only once the sibling
                    # write is recoverable from Saver, as this test promises.
                    import asyncio
                    if not await asyncio.to_thread(sibling_saved.wait, timeout=5):
                        raise AssertionError("The sibling write did not reach the official pending-write checkpoint")
                    raise RuntimeError("process loss after sibling write")
                writes.append(request.tool_call["name"])
                return await handler(request)

        ready, _ = self.invoke(ScriptedModel([preview()]))
        action = self.approved(ready["plan_state"])
        batch = [AIMessage(content="", tool_calls=[
            {"name": "write_file", "args": {"file_path": "/scratch/check.txt", "content": "saved sibling"}, "id": "written", "type": "tool_call"},
            {"name": "read_file", "args": {"file_path": "/scratch/check.txt"}, "id": "uncertain-read", "type": "tool_call"}])]
        model = ScriptedModel([batch, response("read_file", {"file_path": "/scratch/check.txt"}, "retry-read"),
                               answer(), report(), [AIMessage(content="Verified existing file")], grade()])
        saver = PendingWriteSaver()
        graph = self.graph(model, action, saver=saver, memory=binding, store=store, middleware=[CrashRead()])
        config = {"configurable": {"thread_id": "pending-run"}, "recursion_limit": 150}
        context = replace(self.context("pending-run"), memory=binding)
        with self.assertRaisesRegex(RuntimeError, "sibling write") as error:
            self.runner.run(graph.ainvoke({"messages": [HumanMessage(content="Execute", id="pending-user")], "plan_state": ready["plan_state"]},
                         context=context, config=config, durability="sync"))
        self.assertTrue(sibling_saved.is_set())
        finalize_plan_error(graph, config, context, error.exception)
        new_binding = MemoryBinding(scope, store, services)
        self.assertTrue(restore_memory_binding(new_binding, graph.get_state(config).values))
        new_context = replace(self.context("pending-run"), memory=new_binding)
        rebuilt = self.graph(model, action, saver=saver, memory=new_binding, store=store)
        done = self.runner.run(rebuilt.ainvoke(None, context=new_context, config=config, durability="sync"))
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(writes, ["write_file"])
        self.assertEqual(new_context.tool_runtime.execution.tool_call_count, 3)
        receipt = next(message for message in model.calls[1][1] if isinstance(message, ToolMessage) and message.tool_call_id == "uncertain-read")
        self.assertEqual(receipt.status, "error")
        self.assertIn("结果未确认", receipt.content)
        readback = next(message for message in model.calls[2][1] if isinstance(message, ToolMessage) and message.tool_call_id == "retry-read")
        self.assertIn("saved sibling", readback.content)

    def test_structured_public_results_cannot_leak_private_tool_values(self):
        from agent_core.tools import ToolExecutor, ToolRegistry, ToolResult, ToolSpec
        from agent_core_tests.test_runtime import EchoInput
        secret = "opaque-pagination-token"
        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(name="echo_text", description="Read safe tool data",
            input_schema=EchoInput, handler=lambda arguments, runtime: ToolResult(
                content="Observed data, private cursor=" + secret, succeeded=True, sensitive_values=(secret,)))]))
        self.tools = build_langchain_tools(executor.registry.definitions, executor=executor)
        for returned in (secret, "Observed data"):
            with self.subTest(returned=returned):
                scripted = [preview(), response("echo_text", {"text": "hello"}, "private-read"), answer(returned), report(item(result=returned))]
                if returned != secret:
                    scripted += [[AIMessage(content="Observed data")], grade()]
                model = ScriptedModel(scripted)
                ready, _ = self.invoke(model)
                result, _ = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "private-run")
                self.assertNotIn(secret, str(plan_result(result)))
                self.assertNotIn(secret, str(result["plan_state"]))
                if returned == secret:
                    self.assertEqual(result["plan_state"]["phase"], "partial")
                    self.assertNotIn(secret, result["plan_state"]["steps"][0]["result"])
                else:
                    self.assertEqual(result["plan_state"]["phase"], "completed")
                    tool_source = next(value for value in result["plan_state"]["sources"].values() if value["kind"] == "tool")
                    self.assertIn("[redacted]", tool_source["content"])


if __name__ == "__main__":
    unittest.main()
