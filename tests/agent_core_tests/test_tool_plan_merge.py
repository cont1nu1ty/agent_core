"""Discovery/retry contracts combined with upstream plan stages and budgets."""
from dataclasses import replace
import unittest

from agent_core.runtime.budget import ExecutionBudget
from agent_core.tools import ToolExecutor, ToolRegistry, ToolRetryPolicy
from agent_core.tools.contracts import RecoverableToolError
from agent_core.tools.langchain import build_langchain_tools
from agent_core_tests.test_plan_execute import (
    ScriptedModel, preview, response, report, answer, grade,
)


class ToolPlanMergeTest(unittest.TestCase):
    def setUp(self):
        from agent_core_tests.test_plan_execute import PlanExecuteTests
        self.case = PlanExecuteTests()
        self.case.setUp()
        self.addCleanup(self.case.doCleanups)
        original = self.case.context
        def context(*args, **kwargs):
            value = original(*args, **kwargs)
            return replace(value, settings=replace(value.settings, tool_discovery_enabled=True,
                tool_retry_enabled=True, max_input_tokens=None, max_observation_bytes=None),
                tool_runtime=replace(value.tool_runtime, execution=ExecutionBudget(
                    max_tool_calls=24, max_observation_bytes=None)))
        self.case.context = context

    def test_exploration_search_binds_tools_and_preserves_structured_output(self):
        model = ScriptedModel([
            response('search_tools', {'query': 'echo_text'}, 'discover'),
            response('echo_text', {'text': 'observed'}, 'read'), preview(),
        ])
        state, context = self.case.invoke(model, exploration_tools=self.case.tools)
        self.assertEqual(state['plan_state']['phase'], 'ready')
        self.assertNotIn('echo_text', model.bindings_per_call[0])
        self.assertIn('search_tools', model.bindings_per_call[0])
        self.assertIn('ExplorationResult', model.bindings_per_call[0])
        self.assertIn('echo_text', model.bindings_per_call[1])
        self.assertEqual(self.case.tool_calls, ['observed'])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 2)
        self.assertIsNone(context.tool_runtime.execution.max_observation_bytes)

    def test_approved_execution_retries_once_and_report_has_no_business_tools(self):
        ready, _ = self.case.invoke(ScriptedModel([preview()]))
        definition = self.case.executor.registry.definitions[0]
        attempts = []
        def transient(args, runtime):
            attempts.append(args.text)
            if len(attempts) == 1:
                raise ConnectionError('synthetic interruption')
            return definition.handler(args, runtime)
        tool = replace(definition, effect='read', retry_policy=ToolRetryPolicy(mode='safe_read'),
            handler=transient, recoverable_error_mapper=lambda error: RecoverableToolError(
                'connection_unavailable', 'Temporarily unavailable', retryable=True, failure_kind='transient'))
        executor = ToolExecutor(registry=ToolRegistry([tool]))
        self.case.tools = build_langchain_tools([tool], executor=executor)
        model = ScriptedModel([
            response('search_tools', {'query': 'echo_text'}, 'discover'),
            response('echo_text', {'text': 'approved observation'}, 'read'), answer(),
            report(), answer('Observed text 1'), grade(),
        ])
        state, context = self.case.invoke(model, self.case.approved(ready['plan_state']),
            ready['plan_state'], run_id='execute')
        self.assertEqual(state['plan_state']['phase'], 'completed')
        self.assertEqual(attempts, ['approved observation', 'approved observation'])
        self.assertEqual(self.case.tool_calls, ['approved observation'])
        self.assertNotIn('echo_text', model.bindings_per_call[3])
        ledger = context.tool_runtime.execution.snapshot()
        self.assertEqual(ledger['tool_call_count'], 2)
        self.assertEqual(ledger['tool_execution_attempts_by_call']['read'], 2)
        self.assertEqual(ledger['retry_slots_reserved'], 2)

    def test_unlimited_observation_ceiling_restores_without_relaxing_saved_limit(self):
        unlimited = ExecutionBudget(max_observation_bytes=None)
        unlimited.commit_result(None, observation_size=200000)
        restored = ExecutionBudget(max_observation_bytes=None)
        restored.restore(unlimited.snapshot())
        self.assertIsNone(restored.max_observation_bytes)
        self.assertEqual(restored.observation_bytes, 200000)
        limited = ExecutionBudget(max_observation_bytes=100)
        restored.restore({**limited.snapshot(), 'revision': restored.revision + 1})
        self.assertEqual(restored.max_observation_bytes, 100)
        restored.max_observation_bytes = None
        restored.restore(restored.snapshot() | {'limits': limited.snapshot()['limits']})
        self.assertEqual(restored.max_observation_bytes, 100)

    def test_discovery_cannot_expose_tools_outside_exploration_permission(self):
        from langchain_core.messages import ToolMessage
        model = ScriptedModel([
            response('search_tools', {'query': 'echo_text'}, 'discover'), preview(),
        ])
        state, _ = self.case.invoke(model)
        self.assertEqual(state['plan_state']['phase'], 'ready')
        self.assertTrue(all('echo_text' not in tools for tools in model.bindings_per_call))
        receipt = next(message for message in state['messages']
            if isinstance(message, ToolMessage) and message.tool_call_id == 'discover')
        self.assertNotIn('echo_text', str(receipt.content))
        self.assertEqual(self.case.tool_calls, [])

    def test_plan_resume_rejects_mode_change_before_next_stage_model(self):
        from langgraph.checkpoint.memory import InMemorySaver
        from langchain_core.messages import HumanMessage
        from agent_core.contracts.errors import AgentContextError
        from agent_core.planning.plan_execute import PlanInput
        saver = InMemorySaver()
        model = ScriptedModel([RuntimeError('stage interrupted'), preview()])
        context = self.case.context('mode-resume')
        config = {'configurable': {'thread_id': 'mode-resume'}, 'recursion_limit': 150}
        graph = self.case.graph(model, PlanInput(), saver=saver)
        with self.assertRaisesRegex(RuntimeError, 'stage interrupted'):
            self.case.runner.run(graph.ainvoke({'messages': [HumanMessage('Plan', id='user')]}, context=context, config=config))
        for setting in ('tool_discovery_enabled', 'tool_retry_enabled'):
            with self.subTest(setting=setting):
                resumed = self.case.context('mode-resume')
                resumed = replace(resumed, settings=replace(resumed.settings, **{setting: False}))
                rebuilt = self.case.graph(model, PlanInput(), saver=saver)
                before = len(model.calls)
                with self.assertRaisesRegex(AgentContextError, 'Cannot change tool discovery/retry mode'):
                    self.case.runner.run(rebuilt.ainvoke(None, context=resumed, config=config))
                self.assertEqual(len(model.calls), before)

    def test_legacy_plan_document_migrates_without_reusing_old_graph(self):
        from copy import deepcopy
        from agent_core.planning.plan_execute import PLAN_STATE_VERSION, migrate_plan_state
        from agent_core.planning.plan_execute.contracts import parse_plan, PlanActionError
        ready, _ = self.case.invoke(ScriptedModel([preview()]))
        old = deepcopy(ready['plan_state'])
        old['version'] = 'plan-execute-2'
        with self.assertRaises(PlanActionError):
            parse_plan(old)
        migrated = migrate_plan_state(old)
        self.assertEqual(migrated, {**old, 'version': PLAN_STATE_VERSION})
        self.assertEqual(old['version'], 'plan-execute-2')
