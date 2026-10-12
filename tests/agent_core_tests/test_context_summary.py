"""Derived views preserve graph history and use a bounded, cancellable model."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_core.config import AgentConfig
from agent_core.context.summary import ContextSummarizer, input_limit, prepare_context_view
from agent_core.context.budget import select_context
from agent_core.context.checkpoints import ContextDependencyState, read_context_dependencies
from agent_core.contracts.errors import AgentCancelled, AgentContextError, AgentLimitReachedError
from agent_core.contracts.runs import RunRequest
from agent_core.memory.contracts import MemoryScope
from agent_core.models.streaming import stream_chat_message
from agent_core.observability import record_model_call
from agent_core.observability.observers.in_memory import InMemoryUsageObserver
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.planning.plan_execute import PLAN_STATE_VERSION, PlanModelBudget
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.streaming import AsyncStreamRunner
from agent_core_tests.test_runtime import MinimalToolRuntime, ScriptedModel


class ContextSummaryTest(unittest.TestCase):
    def setUp(self):
        self.runner = AsyncStreamRunner()
        self.addCleanup(self.runner.close)
        cancellation = CancellationToken()
        self.archive = []
        scope = MemoryScope("tenant", "user", "project", "session", "task", "summary-run")
        binding = SimpleNamespace(scope=scope, services=SimpleNamespace(),
            history=SimpleNamespace(append=lambda _scope, entries: self.archive.extend(entries)))
        self.context = ReActContext(RunRequest(scope.run_id, "test", "current"), cancellation,
            AgentConfig(context_summary_enabled=True, max_input_tokens=3000, context_summary_trigger_fraction=.1),
            self.runner, lambda: None, MinimalToolRuntime(cancellation), ReActPolicy(), memory=binding)

    def collect(self, model, messages, cancellation, **kwargs):
        kwargs.pop("phase", None)
        kwargs.pop("step_id", None)
        return stream_chat_message(model, messages, cancellation, **kwargs)

    def test_summary_retries_charge_the_shared_plan_budget_and_usage(self):
        budget = PlanModelBudget(max_calls=2)
        context = replace(self.context, model_budget=budget)
        model = ScriptedModel([RuntimeError("provider unavailable"), [AIMessage("Verified summary", usage_metadata={
            "input_tokens": 7, "output_tokens": 3, "total_tokens": 10})]])
        summarizer = ContextSummarizer(model, self.collect, context)
        self.assertEqual(summarizer.summarize([HumanMessage("material", id="source")]), "Verified summary")
        self.assertEqual(budget.calls, 2)
        self.assertEqual(budget.recorded_calls, 2)
        self.assertEqual(budget.unknown_usage_calls, 1)
        self.assertEqual(budget.total_tokens, 10)
        with self.assertRaises(AgentLimitReachedError):
            summarizer.summarize([HumanMessage("more material", id="next")])
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(summarizer.calls, 2)

    def test_unlimited_input_skips_threshold_compaction_but_keeps_provider_window(self):
        settings = replace(self.context.settings, max_input_tokens=None)
        context = replace(self.context, settings=settings, model_budget=PlanModelBudget(max_calls=None))
        model = ScriptedModel([[AIMessage("Bounded summary output")]])
        summarizer = ContextSummarizer(model, self.collect, context)
        messages = [HumanMessage("long history " * 3000, id="old"), HumanMessage("now", id="current")]
        self.assertIsNone(input_limit(settings))
        self.assertTrue(summarizer.fits(messages))
        selected = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[], state={},
            context=context, summarizer=summarizer, archived=True)
        self.assertEqual(selected.messages, messages)
        self.assertEqual(model.calls, [])
        self.assertEqual(summarizer.summarize(messages), "Bounded summary output")
        self.assertEqual(context.model_budget.calls, 1)
        physical = replace(context, settings=replace(settings, model_context_window=3000))
        self.assertEqual(input_limit(physical.settings, 500), 2500)
        self.assertFalse(ContextSummarizer(model, self.collect, physical).fits(messages))

    def test_public_model_command_commits_summary_without_removing_original_messages(self):
        observer = InMemoryUsageObserver()

        def collect(model, messages, cancellation, **kwargs):
            phase = kwargs.pop("phase", "decision")
            return record_model_call(lambda: self.collect(model, messages, cancellation, **kwargs),
                observer=observer, run_id="summary-run", attempt=1, call_id=str(len(observer.records)), phase=phase)

        model = ScriptedModel([[AIMessage("保留早期限制，不写入设备。")], [AIMessage("已核对。")]])
        saver = InMemorySaver()
        graph = compile_react_graph(model, [], stream_chat_message=collect,
                                    policy=self.context.policy, checkpointer=saver)
        messages = [HumanMessage("早期限制 " * 500, id="old-user"), AIMessage("只读", id="old-answer"),
                    HumanMessage("current", id="current-user")]
        config = {"configurable": {"thread_id": "summary-run"}}
        graph.invoke({"messages": messages, "references": [], "planning_rounds": 0,
            "tool_call_count": 0, "observation_bytes": 0, "used_tool_call_ids": []}, config=config, context=self.context)
        saved = graph.get_state(config).values
        self.assertEqual([msg.id for msg in saved["messages"][:3]], [msg.id for msg in messages])
        self.assertEqual(saved["messages"][0].text, messages[0].text)
        self.assertEqual(saved["summary_calls"], 1)
        self.assertEqual(saved["context_summary"]["covered_message_ids"], ["old-user", "old-answer"])
        self.assertIn("保留早期限制", "\n".join(msg.text for msg in model.calls[-1][1]))
        self.assertNotIn(messages[0].text, "\n".join(msg.text for msg in model.calls[-1][1]))
        self.assertEqual([entry.phase for entry in observer.records], ["compaction", "decision"])
        self.assertFalse(any("保留早期限制" in entry.content for entry in self.archive))
        before = len(model.calls)
        graph.invoke(None, config=config, context=self.context)
        self.assertEqual(len(model.calls), before)

    def test_full_prompt_and_tool_schema_trigger_and_current_request_is_preserved(self):
        model = ScriptedModel([[AIMessage("旧约束。")]])
        summarizer = ContextSummarizer(model, self.collect, self.context)
        messages = [HumanMessage("old " * 100, id="old"), AIMessage("answer", id="a"),
                    HumanMessage("不要修改；只有核实后才能发布。", id="current"),
                    AIMessage("", id="pending", tool_calls=[{"name": "read", "id": "p", "args": {}}])]
        prepared = prepare_context_view(messages, system_message=SystemMessage("memory " * 100),
            tools=[{"name": "read", "description": "schema " * 100}], state={}, context=self.context,
            summarizer=summarizer, archived=True)
        self.assertTrue(prepared.changed)
        self.assertIn(messages[2], prepared.messages)
        self.assertIn(messages[3], prepared.messages)
        self.assertNotIn("current", prepared.summary["covered_message_ids"])

    def test_retry_allowance_counts_real_calls_and_failure_retains_original_view(self):
        model = ScriptedModel([RuntimeError("failure")] * 10)
        summarizer = ContextSummarizer(model, self.collect, self.context)
        messages = [HumanMessage("old " * 500, id="old"), AIMessage("answer", id="a"), HumanMessage("now", id="now")]
        for _ in range(3):
            prepared = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[],
                state={}, context=self.context, summarizer=summarizer, archived=True)
            self.assertEqual(prepared.messages, messages)
            self.assertIsNotNone(prepared.warning)
        self.assertEqual(summarizer.calls, 4)
        self.assertEqual(len(model.calls), 4)

    def test_cancel_is_never_swallowed_or_retried(self):
        model = ScriptedModel([AgentCancelled("cancelled")])
        summarizer = ContextSummarizer(model, self.collect, self.context)
        with self.assertRaises(AgentCancelled):
            summarizer.summarize([HumanMessage("material", id="m")])
        self.assertEqual(summarizer.calls, 1)

    def test_oversized_summary_input_is_not_silently_trimmed(self):
        model = ScriptedModel([])
        summarizer = ContextSummarizer(model, self.collect, self.context)
        with self.assertRaises(AgentContextError):
            summarizer.summarize([HumanMessage("material " * 5000, id="m")])
        self.assertEqual(model.calls, [])

    def test_empty_tool_and_overlong_summary_are_rejected_without_retry(self):
        outputs = [AIMessage(""), AIMessage("", tool_calls=[{"id": "bad", "name": "write", "args": {}}]),
                   AIMessage("excess " * 2000)]
        for output in outputs:
            with self.subTest(output=type(output).__name__):
                model = ScriptedModel([[output]])
                summarizer = ContextSummarizer(model, self.collect, self.context)
                with self.assertRaises(AgentContextError):
                    summarizer.summarize([HumanMessage("source material", id="source")])
                self.assertEqual(summarizer.calls, 1)

    def test_without_successful_archive_summary_model_is_never_called(self):
        model = ScriptedModel([])
        messages = [HumanMessage("old " * 3000, id="old"), AIMessage("answer", id="a"), HumanMessage("now", id="now")]
        prepared = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[], state={},
            context=self.context, summarizer=ContextSummarizer(model, self.collect, self.context), archived=False)
        self.assertEqual(prepared.messages, messages)
        self.assertEqual(model.calls, [])

    def test_source_invalidation_discards_saved_summary(self):
        binding = self.context.memory
        binding.services.valid_sources = lambda scope, sources, **kwargs: ()
        context = replace(self.context, settings=replace(self.context.settings, context_summary_enabled=False))
        original = HumanMessage("now", id="current")
        saved = {"text": "obsolete", "source_ids": ["source"], "covered_message_ids": []}
        prepared = prepare_context_view([original], system_message=SystemMessage("system"), tools=[],
            state={"context_summary": saved}, context=context,
            summarizer=ContextSummarizer(ScriptedModel([]), self.collect, context), archived=True)
        self.assertIsNone(prepared.summary)
        self.assertEqual(prepared.messages, [original])
        self.assertTrue(prepared.warning)

    def test_disabled_summary_does_not_call_model_and_complete_group_stays_atomic(self):
        context = replace(self.context, settings=AgentConfig())
        model = ScriptedModel([])
        messages = [HumanMessage("request", id="u"), AIMessage("", id="a",
            tool_calls=[{"name": "read", "id": "read", "args": {}}]), ToolMessage("data", tool_call_id="read", id="t")]
        prepared = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[], state={},
            context=context, summarizer=ContextSummarizer(model, self.collect, context), archived=True)
        self.assertEqual(prepared.messages, messages)
        self.assertEqual(model.calls, [])

    def test_legacy_saved_summary_is_invalidated_after_checkpoint_even_with_live_explicit_sources(self):
        self.context.memory.services.valid_sources = lambda scope, sources, **kwargs: tuple(sources)
        self.context.memory.services.invalidated_sources = lambda scope: ("memory:old:v1",)
        context = replace(self.context, settings=replace(self.context.settings, context_summary_enabled=False))
        original = HumanMessage("now", id="current")
        saved = {"text": "obsolete inferred conclusion", "source_ids": ["old-user"],
                 "covered_message_ids": [], "has_legacy_dependencies": True}
        prepared = prepare_context_view([original], system_message=SystemMessage("system"), tools=[],
            state={"context_summary": saved}, context=context,
            summarizer=ContextSummarizer(ScriptedModel([]), self.collect, context), archived=True)
        self.assertIsNone(prepared.summary)
        self.assertEqual(prepared.messages, [original])
        self.assertTrue(prepared.changed)
        self.assertTrue(prepared.warning)

    def test_tail_dependencies_survive_compaction_and_invalidate_uncovered_history(self):
        invalid = set()
        self.context.memory.services.valid_sources = lambda scope, sources, **kwargs: tuple(
            source for source in sources if source not in invalid)
        messages = [HumanMessage("old " * 500, id="old", additional_kwargs={"history_backfill": True}),
                    AIMessage("inferred prior answer", id="a", additional_kwargs={"history_backfill": True}),
                    HumanMessage("now", id="now")]
        state = {"context_history_dependencies": ["memory:hidden:v1"]}
        prepared = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[],
            state=state, context=self.context,
            summarizer=ContextSummarizer(ScriptedModel([[AIMessage("旧约束。")]]), self.collect, self.context), archived=True)
        self.assertIn("memory:hidden:v1", prepared.summary["source_ids"])
        invalid.add("memory:hidden:v1")
        model = ScriptedModel([])
        invalidated = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[],
            state=state, context=self.context, summarizer=ContextSummarizer(model, self.collect, self.context), archived=True)
        self.assertEqual(invalidated.messages, [messages[-1]])
        self.assertEqual(len(messages), 3)
        self.assertTrue(invalidated.warning)
        self.assertEqual(model.calls, [])

    def test_current_run_unknown_exposure_marks_summary_and_stops_after_invalidation(self):
        messages = [HumanMessage("old " * 500, id="old"), AIMessage("answer", id="a"), HumanMessage("now", id="now")]
        state = {"run_memory_exposure_complete": False}
        first = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[],
            state=state, context=self.context,
            summarizer=ContextSummarizer(ScriptedModel([[AIMessage("旧约束。")]]), self.collect, self.context), archived=True)
        self.assertTrue(first.summary["has_legacy_dependencies"])
        self.context.memory.services.invalidated_sources = lambda scope: ("memory:old:v1",)
        model = ScriptedModel([])
        second = prepare_context_view(messages, system_message=SystemMessage("system"), tools=[],
            state={**state, "context_summary": first.summary}, context=self.context,
            summarizer=ContextSummarizer(model, self.collect, self.context), archived=True)
        self.assertIsNone(second.summary)
        self.assertEqual(model.calls, [])
        self.assertTrue(second.warning)

    def test_final_guard_protects_derived_summary_and_fails_when_required_view_cannot_fit(self):
        summary = AIMessage("early restriction " * 40, id="summary", additional_kwargs={"context_derived": True})
        messages = [summary, HumanMessage("old " * 400, id="old"), AIMessage("old answer", id="a"),
                    HumanMessage("current", id="now")]
        selected = select_context(messages, system_message=SystemMessage("system"), tools=[],
                                  settings=AgentConfig(max_input_tokens=500), archived=True)
        self.assertEqual(selected.messages, [summary, messages[-1]])
        with self.assertRaises(AgentContextError):
            select_context(messages, system_message=SystemMessage("required instructions " * 300), tools=[],
                           settings=AgentConfig(max_input_tokens=500), archived=True)

    def test_dependency_reader_uses_public_graph_state_without_mutating_checkpoint(self):
        from langgraph.graph import StateGraph, START, END
        saver = InMemorySaver()
        builder = StateGraph(ContextDependencyState)
        builder.add_edge(START, END)
        graph = builder.compile(checkpointer=saver)
        config = {"configurable": {"thread_id": "source-run"}}
        identity = {"run_id": "source-run", "user_id": "user", "session_id": "session"}
        graph.invoke({"state_version": "1", "run_identity": identity,
                      "run_exposed_memory_sources": ["memory:record:v1"],
                      "context_history_dependencies": ["prior-run:answer", "memory:inherited:v1"],
                      "context_summary": {"text": "derived context", "has_legacy_dependencies": False,
                                          "source_ids": ["source-run:assistant", "memory:summary:v1"]}}, config=config)
        before = len(list(saver.list(config)))
        self.assertEqual(read_context_dependencies(saver, "source-run", identity=identity),
                         ("memory:record:v1", "prior-run:answer", "memory:inherited:v1", "memory:summary:v1"))
        self.assertEqual(len(list(saver.list(config))), before)
        graph.update_state(config, {"state_version": PLAN_STATE_VERSION})
        self.assertEqual(read_context_dependencies(saver, "source-run", identity=identity),
                         ("memory:record:v1", "prior-run:answer", "memory:inherited:v1", "memory:summary:v1"))
        with self.assertRaises(PermissionError):
            read_context_dependencies(saver, "source-run", identity={"user_id": "another-user"})
        self.assertIsNone(read_context_dependencies(saver, "missing", identity=identity))
        graph.update_state(config, {"run_memory_exposure_complete": False})
        self.assertIsNone(read_context_dependencies(saver, "source-run", identity=identity))
