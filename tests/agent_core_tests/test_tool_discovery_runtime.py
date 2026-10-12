"""Actual create_agent binding, governance and recovery with local Fakes."""

from dataclasses import replace
import json
import unittest

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent_core.tools.contracts import AuthorizationDecision
from agent_core.config import AgentConfig
from agent_core.contracts.context import RunContext
from agent_core.contracts.errors import AgentContextError, AgentUpstreamError
from agent_core.contracts.runs import RunRequest
from agent_core.planning.react.builder import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.planning.react.recovery import prepare_react_resume
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.budget import ExecutionBudget
from agent_core.runtime.service import AgentService
from agent_core.tools import ToolDiscoverySpec, ToolExecutor, ToolRegistry, ToolRetryPolicy, ToolSpec
from agent_core.tools.contracts import AuthorizationDecision, RecoverableToolError
from agent_core.tools.langchain import build_langchain_tools
from agent_core.tools.retry import RetryController
from agent_core.workflows.contracts import WorkflowDefinition, WorkflowExecution
from agent_core.workflows.registry import WorkflowRegistry
from agent_core_tests.test_runtime import EchoInput, MinimalToolRuntime, ScriptedModel


def call(name, identity, **args):
    return {"name": name, "id": identity, "args": args, "type": "tool_call"}


def decision(*calls):
    return [AIMessage(content="", tool_calls=list(calls))]


class CaptureModel(ScriptedModel):
    def __init__(self, responses):
        super().__init__(responses)
        self.bindings = []
        self.schemas = []

    def bind_tools(self, tools, **kwargs):
        self.bindings.append([tool.name for tool in tools])
        self.schemas.append([convert_to_openai_tool(tool) for tool in tools])
        return super().bind_tools(tools, **kwargs)


def service(model, definitions, *, saver=None, settings=None, contexts=None, executor=None, native_tools=()):
    executor = executor or ToolExecutor(registry=ToolRegistry(definitions))
    def factory(request, context, resources):
        runtime = MinimalToolRuntime(context.cancellation, execution=ExecutionBudget(max_tool_calls=resources.settings.max_tool_calls,
            max_observation_bytes=resources.settings.max_observation_bytes))
        react = ReActContext(request, context.cancellation, resources.settings,
                             resources.async_stream_runner, context.clock, runtime, ReActPolicy())
        if contexts is not None:
            contexts.append(react)
        return WorkflowExecution(compile_react_graph(resources.model,
            [*build_langchain_tools(definitions, executor=executor), *native_tools], stream_chat_message=resources.stream_chat_message,
            policy=react.policy, checkpointer=resources.checkpointer,
            native_authorizer=lambda **_: AuthorizationDecision(True)),
            {"messages": [HumanMessage(request.input)], "references": [], "planning_rounds": 0,
             "used_tool_call_ids": [], "tool_call_count": 0, "observation_bytes": 0},
            context=react, state_version="2", prepare_resume=prepare_react_resume)
    return AgentService(workflows=WorkflowRegistry([WorkflowDefinition("discovery-test", factory)]),
        settings=settings or AgentConfig(tool_discovery_enabled=True), model_factory=lambda: model, checkpointer=saver)


def run(agent, request=None):
    return list(agent.stream(request or RunRequest("tool-run", "discovery-test", "read values"),
                             RunContext(CancellationToken())))


def specification(name, calls, *, exposure="deferred", maximum=8, handler=None):
    def echo(args, _runtime):
        calls.append((name, args.text))
        return args.text
    return ToolSpec(name=name, description=f"Read a named value using {name}", input_schema=EchoInput,
        handler=handler or echo, max_calls_per_run=maximum, effect="read",
        discovery=ToolDiscoverySpec(exposure=exposure, aliases=(name,)))


class ToolDiscoveryRuntimeTest(unittest.TestCase):
    def test_discovery_budget_configuration_requires_positive_integers(self):
        for name in ("max_discovery_calls", "max_tool_schema_tokens"):
            for value in (None, True, 0, -1, 1.5):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    AgentConfig(**{name: value})

    def test_search_binds_next_request_and_cannot_authorize_same_batch(self):
        executed = []
        definitions = [specification("read_value", executed), specification("secret_value", executed, exposure="hidden")]
        model = CaptureModel([
            decision(call("search_tools", "search1", query="read_value"), call("read_value", "guess1", text="guessed")),
            decision(call("read_value", "read1", text="accepted")), [AIMessage(content="done")]])
        contexts = []
        run(service(model, definitions, contexts=contexts))
        self.assertEqual(model.bindings[0], ["search_tools"])
        self.assertIn("read_value", model.bindings[1])
        parameters = next(item["function"]["parameters"] for item in model.schemas[1]
                          if item["function"]["name"] == "read_value")
        self.assertEqual(parameters["properties"]["text"]["type"], "string")
        self.assertEqual(parameters["required"], ["text"])
        self.assertNotIn("runtime", parameters["properties"])
        self.assertTrue(all("secret_value" not in binding for binding in model.bindings))
        self.assertEqual(executed, [("read_value", "accepted")])
        errors = [json.loads(message.text) for message in model.calls[1][1]
                  if isinstance(message, ToolMessage) and message.status == "error"]
        self.assertEqual(errors[0]["errorCode"], "tool_not_exposed")
        self.assertEqual(contexts[-1].tool_runtime.execution.calls_by_category["discovery"], 1)

    def test_two_searches_accumulate_despite_tool_send_state_snapshots(self):
        executed = []
        definitions = [specification(name, executed) for name in ("read_alpha", "read_beta")]
        model = CaptureModel([
            decision(call("search_tools", "s1", query="read_alpha", limit=1),
                     call("search_tools", "s2", query="read_beta", limit=1)),
            decision(call("read_alpha", "a", text="A"), call("read_beta", "b", text="B")),
            [AIMessage(content="done")]])
        run(service(model, definitions))
        self.assertTrue({"read_alpha", "read_beta"}.issubset(model.bindings[1]))
        self.assertEqual(executed, [("read_alpha", "A"), ("read_beta", "B")])

    def test_discovery_exhaustion_does_not_disable_loaded_business_tool(self):
        executed = []
        model = CaptureModel([
            decision(*(call("search_tools", f"s{i}", query="read_value") for i in range(3))),
            decision(call("read_value", "read", text="ok")), [AIMessage(content="done")]])
        contexts = []
        run(service(model, [specification("read_value", executed)], contexts=contexts))
        self.assertNotIn("search_tools", model.bindings[1])
        self.assertIn("read_value", model.bindings[1])
        self.assertEqual(executed, [("read_value", "ok")])
        self.assertIsNone(contexts[-1].tool_runtime.execution.stop_reason)

    def test_single_tool_exhaustion_keeps_other_tools_available(self):
        executed = []
        definitions = [specification("read_once", executed, exposure="direct", maximum=1),
                       specification("read_other", executed, exposure="direct")]
        model = CaptureModel([decision(call("read_once", "one", text="one")),
                              decision(call("read_other", "two", text="two")), [AIMessage(content="done")]])
        run(service(model, definitions))
        self.assertNotIn("read_once", model.bindings[1])
        self.assertIn("read_other", model.bindings[1])
        self.assertEqual(len(executed), 2)

    def test_partial_search_checkpoint_survives_sibling_failure(self):
        saver = InMemorySaver()
        executed = []
        crashes = []
        def unstable(args, _runtime):
            crashes.append(args.text)
            if len(crashes) == 1:
                raise RuntimeError("test interruption")
            return "recovered"
        definitions = [specification("read_value", executed),
                       specification("unstable_read", [], exposure="direct", handler=unstable)]
        request = RunRequest("search-recovery", "discovery-test", "read")
        first = CaptureModel([decision(call("search_tools", "s1", query="read_value"),
                                       call("unstable_read", "u1", text="first"))])
        with self.assertRaises(AgentUpstreamError):
            run(service(first, definitions, saver=saver), request)
        resumed = CaptureModel([decision(call("read_value", "r1", text="after recovery")), [AIMessage(content="done")]])
        run(service(resumed, definitions, saver=saver), replace(request, resume=True, attempt=2))
        self.assertIn("read_value", resumed.bindings[0])
        self.assertEqual(executed, [("read_value", "after recovery")])
        self.assertEqual(len([m for m in resumed.calls[0][1] if isinstance(m, ToolMessage) and m.tool_call_id == "s1"]), 1)

    def test_changed_discovered_schema_refuses_resume(self):
        saver = InMemorySaver()
        attempts = []
        def fail(_args, _runtime):
            attempts.append("failed read")
            raise RuntimeError("interrupt")
        definitions = [specification("read_value", []), specification("unstable_read", [], exposure="direct", handler=fail)]
        request = RunRequest("changed", "discovery-test", "read")
        # Commit discovery in its own superstep before injecting a failure.
        # Same-batch pending-write recovery is covered separately above.
        first = CaptureModel([decision(call("search_tools", "s1", query="read_value")),
                              decision(call("unstable_read", "u1", text="fail"))])
        with self.assertRaises(AgentUpstreamError):
            run(service(first, definitions, saver=saver), request)
        self.assertIn("read_value", first.bindings[1])
        self.assertEqual(attempts, ["failed read"])
        changed = [replace(definitions[0], version="2.0.0"), definitions[1]]
        with self.assertRaises(AgentContextError) as caught:
            run(service(CaptureModel([]), changed, saver=saver), replace(request, resume=True, attempt=2))
        self.assertEqual(caught.exception.code, "tool_catalog_changed")
        self.assertEqual(attempts, ["failed read"])

    def test_retry_reservation_survives_crash_and_old_call_is_not_replayed(self):
        saver = InMemorySaver()
        observed = []
        class Transient(Exception):
            pass
        def flaky(args, _runtime):
            observed.append(args.text)
            if len(observed) == 1:
                raise Transient()
            if len(observed) == 2:
                raise RuntimeError("interrupted during retry")
            return "fresh success"
        definition = replace(specification("read_value", [], exposure="direct", handler=flaky),
            retry_policy=ToolRetryPolicy(mode="safe_read"),
            recoverable_error_mapper=lambda error: RecoverableToolError("temporary", "temporary", True, "transient")
                if isinstance(error, Transient) else None)
        clock = [0.0]
        controller = RetryController(wait=lambda seconds: clock.__setitem__(0, clock[0] + seconds),
                                     jitter=lambda *_: 1.0, monotonic=lambda: clock[0])
        executor = ToolExecutor(registry=ToolRegistry([definition]), retry_controller=controller)
        settings = AgentConfig(tool_retry_enabled=True, max_tool_calls=24, max_external_tool_calls=16)
        request = RunRequest("retry-recovery", "discovery-test", "read")
        with self.assertRaises(AgentUpstreamError):
            run(service(CaptureModel([decision(call("read_value", "first", text="old"))]), [definition],
                        saver=saver, settings=settings, executor=executor), request)
        contexts = []
        resumed = CaptureModel([decision(call("read_value", "second", text="new")), [AIMessage(content="done")]])
        run(service(resumed, [definition], saver=saver,
                    settings=replace(settings, max_tool_calls=64, max_external_tool_calls=32, max_tool_retry_slots=64),
                    contexts=contexts, executor=executor),
            replace(request, resume=True, attempt=2))
        self.assertEqual(observed, ["old", "old", "new"])
        interrupted = [json.loads(m.text) for m in resumed.calls[0][1] if isinstance(m, ToolMessage)]
        self.assertEqual(interrupted[0]["errorCode"], "tool_execution_interrupted")
        budget = contexts[-1].tool_runtime.execution
        self.assertEqual(budget.retry_reserved_count, 4)
        self.assertEqual(budget.call_reservations["first"]["outcome"], "unknown")
        self.assertIsNone(budget.call_reservations["first"]["handler_attempts"])
        self.assertEqual(budget.calls_by_tool["read_value"], 2)
        self.assertEqual(budget.max_tool_calls, 24)
        self.assertEqual(budget.max_external_tool_calls, 16)
        self.assertEqual(budget.max_retry_attempts, 8)

    def test_revoked_permission_cannot_be_regranted_within_same_run(self):
        allowed = [True]
        executed = []
        class DynamicAuthorizer:
            def authorize(self, definition, _runtime):
                return AuthorizationDecision(definition.name != "read_value" or allowed[0], "denied", "denied")
        def revoke(_args, _runtime):
            allowed[0] = False
            return "revoked"
        def grant(_args, _runtime):
            allowed[0] = True
            return "granted for future runs"
        definitions = [specification("read_value", executed, exposure="direct"),
            specification("revoke", [], exposure="direct", handler=revoke),
            specification("grant", [], exposure="direct", handler=grant)]
        model = CaptureModel([decision(call("revoke", "r", text=""), call("read_value", "denied1", text=""),
            call("grant", "g", text=""), call("read_value", "denied2", text="")), [AIMessage(content="done")]])
        run(service(model, definitions, executor=ToolExecutor(registry=ToolRegistry(definitions), authorizer=DynamicAuthorizer())))
        self.assertEqual(executed, [])
        self.assertNotIn("read_value", model.bindings[1])
        errors = [json.loads(message.text)["errorCode"] for message in model.calls[1][1]
                  if isinstance(message, ToolMessage) and message.status == "error"]
        self.assertEqual(errors, ["tool_not_allowed", "tool_not_allowed"])

    def test_native_contribution_cannot_grant_itself_protected_state_access(self):
        @tool
        def unsafe_native(runtime: ToolRuntime) -> Command:
            """A native tool with an invalid attempt to change governance state."""
            return Command(update={"messages": [ToolMessage("done", tool_call_id=runtime.tool_call_id)],
                                   "tool_discovery": {"revision": 999, "discovered": ["secret"]}})
        unsafe_native.extras = {"agent_core_catalog": {
            "version": "1", "category": "test", "source": "test", "budget_group": "external", "effect": "control",
            "discovery": ToolDiscoverySpec(exposure="direct"), "allowed_state_keys": ("tool_discovery",)}}
        model = CaptureModel([decision(call("unsafe_native", "native"))])
        with self.assertRaises(AgentContextError):
            run(service(model, [], native_tools=(unsafe_native,)))

    def test_permission_revoked_between_retries_stays_revoked_after_grant(self):
        allowed, executed = [True], []
        class Transient(Exception):
            pass
        class DynamicAuthorizer:
            def authorize(self, definition, _runtime):
                return AuthorizationDecision(definition.name != "read_value" or allowed[0], "denied", "denied")
        def flaky(_args, _runtime):
            executed.append("first attempt")
            allowed[0] = False
            raise Transient()
        def grant(_args, _runtime):
            allowed[0] = True
            return "granted for future runs"
        definition = replace(specification("read_value", [], exposure="direct", handler=flaky),
            retry_policy=ToolRetryPolicy(mode="safe_read"),
            recoverable_error_mapper=lambda _: RecoverableToolError("temporary", "temporary", True, "transient"))
        definitions = [definition, specification("grant", [], exposure="direct", handler=grant)]
        elapsed = [0.0]
        controller = RetryController(wait=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
                                     jitter=lambda *_: 1.0, monotonic=lambda: elapsed[0])
        model = CaptureModel([decision(call("read_value", "read", text=""), call("grant", "grant", text="")),
                              [AIMessage(content="done")]])
        contexts = []
        executor = ToolExecutor(registry=ToolRegistry(definitions), authorizer=DynamicAuthorizer(), retry_controller=controller)
        run(service(model, definitions, settings=AgentConfig(tool_discovery_enabled=True, tool_retry_enabled=True),
                    contexts=contexts, executor=executor))
        self.assertEqual(executed, ["first attempt"])
        self.assertNotIn("read_value", model.bindings[1])
        self.assertIn("read_value", contexts[-1].tool_runtime.execution.snapshot()["revoked_tool_names"])

    def test_resume_cannot_switch_feature_modes_in_either_direction(self):
        def fail(_args, _runtime):
            raise RuntimeError("interrupt before result")
        definitions = [specification("unstable_read", [], exposure="direct", handler=fail)]
        for flag in ("tool_discovery_enabled", "tool_retry_enabled"):
            for enabled in (False, True):
                with self.subTest(flag=flag, enabled=enabled):
                    saver = InMemorySaver()
                    request = RunRequest(f"switch-{flag}-{enabled}", "discovery-test", "read")
                    settings = replace(AgentConfig(), **{flag: enabled})
                    with self.assertRaises(AgentUpstreamError):
                        run(service(CaptureModel([decision(call("unstable_read", "u", text=""))]), definitions,
                                    saver=saver, settings=settings), request)
                    with self.assertRaises(AgentContextError):
                        run(service(CaptureModel([]), definitions, saver=saver,
                            settings=replace(settings, **{flag: not enabled})), replace(request, resume=True, attempt=2))


if __name__ == "__main__":
    unittest.main()
