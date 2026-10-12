"""Durable public create_agent composition and complete-input selection."""

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command, interrupt

from agent_core.tools.contracts import AuthorizationDecision
from agent_core.config import AgentConfig
from agent_core.context.budget import archive_messages, select_context
from agent_core.context.history import ToolHistoryMiddleware
from agent_core.contracts.context import RunContext
from agent_core.contracts.errors import AgentContextError, AgentInputBudgetError, AgentUpstreamError
from agent_core.contracts.runs import RunRequest
from agent_core.memory.contracts import HistoryEntry
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.planning.react.recovery import prepare_react_resume
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.service import AgentService
from agent_core.tools import ToolExecutor, ToolRegistry, ToolSpec, ToolResult
from agent_core.tools.langchain import build_langchain_tools
from agent_core.workflows.contracts import WorkflowDefinition, WorkflowExecution
from agent_core.workflows.registry import WorkflowRegistry
from agent_core_tests.test_runtime import EchoInput, MinimalToolRuntime, ScriptedModel
from agent_core_tests.test_runtime import GenericEvidence


def service_for(model, saver, tools, contexts, *, middleware=()):
    def factory(request, context, resources):
        runtime = MinimalToolRuntime(context.cancellation)
        react = ReActContext(request, context.cancellation, resources.settings,
                             resources.async_stream_runner, context.clock, runtime, ReActPolicy())
        contexts.append(react)
        return WorkflowExecution(
            compile_react_graph(resources.model, tools, stream_chat_message=resources.stream_chat_message,
                                policy=react.policy, checkpointer=resources.checkpointer, middleware=middleware,
                                native_authorizer=lambda **_: AuthorizationDecision(True)),
            {"messages": [HumanMessage(request.input)], "references": [], "planning_rounds": 0,
             "used_tool_call_ids": [], "tool_call_count": 0, "observation_bytes": 0}, context=react,
            prepare_resume=prepare_react_resume,
        )
    return AgentService(workflows=WorkflowRegistry([WorkflowDefinition("test", factory)]),
                        model_factory=lambda: model, checkpointer=saver)


class RecoveryTest(unittest.TestCase):
    def test_replayed_tool_preserves_both_safe_observations_after_uncommitted_checkpoint(self):
        class Archive:
            def __init__(self):
                self.entries = {}

            def append(self, scope, entries):
                for entry in entries:
                    old = self.entries.get(entry.source_id)
                    if old is not None and old.content != entry.content:
                        raise ValueError("cannot overwrite original history")
                    self.entries[entry.source_id] = entry

        archive = Archive()
        binding = SimpleNamespace(scope=SimpleNamespace(run_id="replayed"), history=archive)
        user = HumanMessage("read current state", id="user")
        hook = ToolHistoryMiddleware()
        results = []
        # First execution reached History, then its process was lost before a
        # graph checkpoint. The same outstanding read runs in host attempt 2.
        for attempt, text in ((1, "state before crash"), (2, "state after restart")):
            request = SimpleNamespace(tool_call={"id": "read-one", "name": "read_state"},
                state={"messages": [user]}, runtime=SimpleNamespace(context=SimpleNamespace(
                    memory=binding, request=RunRequest("replayed", "test", "read", attempt=attempt))))
            results.append(hook.wrap_tool_call(request,
                lambda _request, value=text: ToolMessage(value, tool_call_id="read-one")))
        sources = [message.additional_kwargs["history_source_id"] for message in results]
        self.assertNotEqual(*sources)
        self.assertEqual([archive.entries[source].content for source in sources],
                         ["state before crash", "state after restart"])
        self.assertEqual({archive.entries[source].call_id for source in sources}, {"read-one"})

    def test_partial_tool_batch_resumes_completed_pending_writes_without_repeating(self):
        saver = InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=[
            ("agent_core_tests.test_runtime", "GenericEvidence")]))
        calls = []
        failed = False

        def echo(args, runtime):
            nonlocal failed
            calls.append(args.text)
            if args.text == "second" and not failed:
                failed = True
                raise RuntimeError("simulated process interruption")
            return ToolResult(args.text, succeeded=True, reference=GenericEvidence(runtime.execution.next_reference_id()),
                              sensitive_values=("private-value",))

        executor = ToolExecutor(registry=ToolRegistry([ToolSpec(
            name="echo_text", description="Echo", input_schema=EchoInput, handler=echo)]))
        tools = build_langchain_tools(executor.registry.definitions, executor=executor)
        contexts = []
        first = ScriptedModel([[AIMessage(content="", tool_calls=[
            {"name": "echo_text", "args": {"text": text}, "id": text, "type": "tool_call"}
            for text in ("first", "second")])]])
        request = RunRequest("recovery-run", "test", "perform both")
        with self.assertRaises(AgentUpstreamError):
            list(service_for(first, saver, tools, contexts).stream(request, RunContext(CancellationToken())))
        self.assertEqual(calls, ["first", "second"])
        second = ScriptedModel([[AIMessage(content="done")]])
        events = list(service_for(second, saver, tools, contexts).stream(
            replace(request, resume=True, attempt=2), RunContext(CancellationToken())))
        self.assertEqual(calls, ["first", "second", "second"])
        self.assertEqual(contexts[-1].tool_runtime.execution.tool_call_count, 2)
        self.assertEqual(contexts[-1].tool_runtime.execution.observation_bytes, len("firstsecond"))
        self.assertEqual([item.reference_id for item in contexts[-1].tool_runtime.execution.references], ["R1", "R2"])
        self.assertEqual(contexts[-1].tool_runtime.execution.sensitive_values_snapshot(), ("private-value",))
        result = events[-1].payload["result"]
        self.assertEqual(result.output, "done")
        self.assertEqual(result.metadata["checkpoint"]["thread_id"], request.run_id)
        sent = second.calls[0][1]
        self.assertEqual(sum(isinstance(msg, HumanMessage) for msg in sent), 1)
        self.assertEqual([msg.tool_call_id for msg in sent if isinstance(msg, ToolMessage)], ["first", "second"])

    def test_completed_run_resume_returns_saved_result_without_model_call(self):
        saver = InMemorySaver()
        request = RunRequest("finished", "test", "answer")
        original = list(service_for(ScriptedModel([[AIMessage(content="saved answer")]]), saver, [], []).stream(
            request, RunContext(CancellationToken())))
        model = ScriptedModel([])
        resumed = list(service_for(model, saver, [], []).stream(replace(request, resume=True), RunContext(CancellationToken())))
        self.assertEqual(resumed[-1].payload["result"].output, original[-1].payload["result"].output)
        self.assertEqual(model.calls, [])

    def test_missing_checkpoint_fails_instead_of_rerunning(self):
        model = ScriptedModel([])
        with self.assertRaisesRegex(AgentContextError, "missing"):
            list(service_for(model, InMemorySaver(), [], []).stream(
                RunRequest("missing", "test", "answer", resume=True), RunContext(CancellationToken())))
        self.assertEqual(model.calls, [])

    def test_interrupt_is_not_completion(self):
        @tool
        def pause() -> str:
            """Pause this test graph."""
            return interrupt("test-only pause")

        model = ScriptedModel([[AIMessage(content="", tool_calls=[
            {"name": "pause", "args": {}, "id": "pause1", "type": "tool_call"}])]])
        events = list(service_for(model, InMemorySaver(), [pause], []).stream(
            RunRequest("paused", "test", "pause"), RunContext(CancellationToken())))
        self.assertEqual(events[-1].kind, "run.suspended")
        self.assertFalse(any(event.kind == "run.completed" for event in events))

    def test_native_command_cannot_change_execution_budget(self):
        @tool
        def tamper(runtime: ToolRuntime[ReActContext]) -> Command:
            """An intentionally invalid extension."""
            return Command(update={"tool_call_count": 0,
                                   "messages": [ToolMessage("bad", tool_call_id=runtime.tool_call_id)]})

        model = ScriptedModel([[AIMessage(content="", tool_calls=[
            {"name": "tamper", "args": {}, "id": "bad", "type": "tool_call"}])]])
        with self.assertRaisesRegex(AgentContextError, "protected"):
            list(service_for(model, None, [tamper], []).stream(
                RunRequest("bad", "test", "test"), RunContext(CancellationToken())))


class ContextBudgetTest(unittest.TestCase):
    def test_disabled_application_cap_keeps_and_counts_complete_archived_input(self):
        messages = [HumanMessage("Old request " * 8000, id="old-user"),
                    AIMessage("Old answer", id="old-answer"),
                    HumanMessage("Current request", id="user"),
                    AIMessage(content="", id="call", tool_calls=[{"name": "read_data", "id": "read", "args": {}}]),
                    ToolMessage("evidence " * 8000, id="result", name="read_data", tool_call_id="read")]
        archived_entries = []
        binding = SimpleNamespace(scope=SimpleNamespace(run_id="run"),
                                  history=SimpleNamespace(append=lambda scope, entries: archived_entries.extend(entries)))
        sources = archive_messages(binding, messages)
        self.assertEqual(len(sources), len(messages))
        self.assertEqual([entry.content for entry in archived_entries], [message.text for message in messages])

        def unnecessary_readback(source):
            raise AssertionError("A disabled input cap must not compact archived observations")

        for archived in (False, True):
            with self.subTest(archived=archived):
                selected = select_context(messages, system_message=SystemMessage("Required instructions " * 1000),
                    tools=[{"name": "read_data", "description": "Tool schema " * 1000}],
                    settings=AgentConfig(max_input_tokens=None), archived=archived,
                    tool_sources={"result": sources[-1]}, history_reader=unnecessary_readback)
                self.assertIsNone(selected.input_limit)
                self.assertGreater(selected.input_tokens, AgentConfig().max_input_tokens)
                self.assertEqual(selected.counting, "estimated")
                self.assertEqual(selected.messages, messages)
                self.assertEqual(selected.removed_messages, 0)
                self.assertEqual(selected.compacted_messages, 0)
                self.assertGreater(selected.input_tokens, select_context(messages,
                    system_message=SystemMessage("Required"), tools=[],
                    settings=AgentConfig(max_input_tokens=None), archived=archived).input_tokens)

    def test_disabled_application_cap_still_respects_explicit_model_window(self):
        settings = AgentConfig(max_input_tokens=None, model_context_window=1000, output_token_reserve=100)
        selected = select_context([HumanMessage("short request")], system_message=SystemMessage("Required"),
                                  tools=[], settings=settings, archived=False)
        self.assertEqual(selected.input_limit, 900)
        self.assertLessEqual(selected.input_tokens, selected.input_limit)
        with self.assertRaises(AgentInputBudgetError) as caught:
            select_context([HumanMessage("required input " * 100)], system_message=SystemMessage("Required"),
                           tools=[], settings=settings, archived=False, output_tokens=900)
        self.assertEqual(caught.exception.input_limit, 100)
        self.assertGreater(caught.exception.input_tokens, 100)
        with self.assertRaisesRegex(AgentContextError, "leaves no input budget"):
            select_context([HumanMessage("request")], system_message=SystemMessage("Required"),
                           tools=[], settings=settings, archived=True, output_tokens=1000)

    def test_last_completed_read_page_uses_a_recoverable_literal_window(self):
        observation = "0123456789" * 1600
        messages = [HumanMessage("Keep confirmed constraints", id="user"),
                    AIMessage(content="", tool_calls=[{"name": "read_file", "id": "read", "args": {"limit": 100}}]),
                    ToolMessage(observation, id="result", name="read_file", tool_call_id="read")]
        result = select_context(messages, system_message=SystemMessage("Required instructions " * 250),
                                tools=[], settings=AgentConfig(max_input_tokens=2400), archived=True,
                                tool_sources={"result": "run:result"},
                                history_reader=lambda source: HistoryEntry(source, "run", 1, "tool", observation))
        self.assertLessEqual(result.input_tokens, 2400)
        self.assertEqual(result.compacted_messages, 1)
        self.assertEqual(result.removed_messages, 0)
        self.assertEqual(result.messages[:2], messages[:2])
        window = result.messages[-1]
        page = json.loads(window.content.splitlines()[1])
        self.assertEqual(page["sourceId"], "run:result")
        self.assertTrue(page["truncated"])
        self.assertEqual(page["nextOffset"], page["returnedCharacters"])
        self.assertEqual(window.content.splitlines()[-1], observation[:page["returnedCharacters"]])
        self.assertEqual(messages[-1].content, observation)
        self.assertEqual(window.tool_call_id, "read")

    def test_no_readback_or_unfinished_batch_does_not_discard_required_observations(self):
        messages = [HumanMessage("Keep confirmed constraints", id="user"),
                    AIMessage(content="", tool_calls=[{"name": "read_file", "id": "read", "args": {}}]),
                    ToolMessage("evidence" * 2000, id="result", name="read_file", tool_call_id="read")]
        kwargs = {"system_message": SystemMessage("required"), "tools": [],
                  "settings": AgentConfig(max_input_tokens=1000), "archived": True}
        with self.assertRaises(AgentInputBudgetError):
            select_context(messages, **kwargs)
        messages[1] = AIMessage(content="", tool_calls=[*messages[1].tool_calls,
                                {"name": "other", "id": "unfinished", "args": {}}])
        with self.assertRaises(AgentInputBudgetError):
            select_context(messages, tool_sources={"result": "run:result"},
                           history_reader=lambda source: HistoryEntry(source, "run", 1, "tool", messages[-1].content), **kwargs)

    def test_window_offsets_follow_canonical_history_not_offload_preview_or_raw_json(self):
        raw = '{ "data": "' + "SAFE " * 3000 + '", "private_cursor": "PRIVATE_VALUE" }'
        canonical = json.dumps({"data": "SAFE " * 3000}, separators=(",", ":"))
        messages = [HumanMessage("Read the saved data"),
                    AIMessage(content="", tool_calls=[{"name": "read_data", "id": "read", "args": {}}]),
                    ToolMessage(raw, id="result", name="read_data", tool_call_id="read")]
        reads = []

        def read(source):
            reads.append(source)
            return HistoryEntry(source, "run", 1, "tool", canonical,
                                metadata={"offset_characters": 0, "original_characters": len(canonical)})

        selected = select_context(messages, system_message=SystemMessage("required"), tools=[],
                                  settings=AgentConfig(max_input_tokens=1000), archived=True,
                                  tool_sources={"result": "run:original"}, history_reader=read)
        content = selected.messages[-1].content
        page = json.loads(content.splitlines()[1])
        self.assertEqual(reads, ["run:original"])
        self.assertEqual(content.splitlines()[-1], canonical[:page["nextOffset"]])
        self.assertEqual(page["totalCharacters"], len(canonical))
        self.assertNotIn("PRIVATE_VALUE", content)
        self.assertEqual(messages[-1].content, raw)
        messages[-1] = messages[-1].model_copy(update={"name": "history_read"})
        with self.assertRaises(AgentInputBudgetError):
            select_context(messages, system_message=SystemMessage("required"), tools=[],
                           settings=AgentConfig(max_input_tokens=1000), archived=True,
                           tool_sources={"result": "run:original"}, history_reader=read)
        self.assertEqual(reads, ["run:original"])

    def test_tools_and_system_memory_count_even_for_a_short_question(self):
        settings = AgentConfig(max_input_tokens=30)
        with self.assertRaises(AgentContextError):
            select_context([HumanMessage("hi")], system_message=SystemMessage("memory" * 100),
                           tools=[], settings=settings, archived=True)
        with self.assertRaises(AgentContextError):
            select_context([HumanMessage("hi")], system_message=SystemMessage("system"),
                           tools=[{"name": "huge", "description": "schema" * 100}], settings=settings, archived=True)

    def test_archive_is_required_and_tool_exchange_is_removed_as_a_unit(self):
        messages = [HumanMessage("old " * 100), AIMessage(content="old answer"), HumanMessage("current"),
                    AIMessage(content="", tool_calls=[{"name": "tool", "id": "id", "args": {}}]),
                    ToolMessage("current evidence", tool_call_id="id")]
        settings = AgentConfig(max_input_tokens=100)
        with self.assertRaisesRegex(AgentContextError, "archive"):
            select_context(messages, system_message=SystemMessage("system"), tools=[], settings=settings, archived=False)
        result = select_context(messages, system_message=SystemMessage("system"), tools=[], settings=settings, archived=True)
        self.assertEqual(result.messages, messages[2:])
        self.assertEqual(result.removed_messages, 2)

    def test_model_window_includes_explicit_output_reservation(self):
        with self.assertRaises(AgentContextError):
            select_context([HumanMessage("hello" * 20)], system_message=SystemMessage("system"), tools=[],
                           settings=AgentConfig(model_context_window=100), archived=True, output_tokens=90)


if __name__ == "__main__":
    unittest.main()
