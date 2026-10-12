"""Async-only resources exercise real graph routing, pauses and cancellation."""
import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace
from typing import TypedDict
import unittest

from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.store.memory import InMemoryStore

from agent_core import (
    AgentService, CancellationToken, PersistenceResources, ResumeDecision,
    RunContext, RunRequest, WorkflowDefinition, WorkflowExecution, WorkflowRegistry,
)
from agent_core.contracts.errors import AgentCancelled, AgentContextError
from agent_core.memory import MemoryBinding, MemoryScope, build_memory_components
from agent_core.memory.contracts import MemoryRelease, MemorySnapshot
from agent_core.planning.react import compile_react_graph
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.tools.contracts import AuthorizationDecision, ToolDiscoverySpec
from agent_core_tests.test_runtime import MinimalToolRuntime


class AsyncOnlySaver(InMemorySaver):
    def __init__(self):
        super().__init__()
        self.loops = []

    def get_tuple(self, *_args, **_kwargs):
        raise AssertionError("sync checkpoint read")

    def put(self, *_args, **_kwargs):
        raise AssertionError("sync checkpoint write")

    def put_writes(self, *_args, **_kwargs):
        raise AssertionError("sync pending write")

    async def aget_tuple(self, config):
        self.loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return InMemorySaver.get_tuple(self, config)

    async def aput(self, config, checkpoint, metadata, new_versions):
        self.loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return InMemorySaver.put(self, config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):
        self.loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        return InMemorySaver.put_writes(self, config, writes, task_id, task_path)


class AsyncOnlyStore(InMemoryStore):
    def batch(self, *_args, **_kwargs):
        raise AssertionError("sync store access")

    async def abatch(self, ops):
        await asyncio.sleep(0)
        return InMemoryStore.batch(self, ops)


class AsyncModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.loops, self.closed = [], 0

    def bind_tools(self, *_args, **_kwargs):
        return self

    def stream(self, *_args, **_kwargs):
        raise AssertionError("sync model access")

    async def astream(self, *_args, **_kwargs):
        self.loops.append(asyncio.get_running_loop())
        await asyncio.sleep(0)
        yield self.responses.pop(0)

    async def aclose(self):
        self.loops.append(asyncio.get_running_loop())
        self.closed += 1


def native_service(model, native_tool, saver, *, authorizer=None, middleware=(), ownership="owned"):
    def factory(request, context, resources):
        policy = ReActPolicy()
        runtime = ReActContext(request, context.cancellation, resources.settings,
            None, context.clock, MinimalToolRuntime(context.cancellation), policy)
        graph = compile_react_graph(resources.model, [native_tool], policy=policy,
            stream_chat_message=resources.astream_chat_message, checkpointer=resources.checkpointer,
            native_authorizer=authorizer, middleware=middleware)
        return WorkflowExecution(graph, {"messages": [HumanMessage(request.input)]}, context=runtime)
    return AgentService(workflows=WorkflowRegistry([WorkflowDefinition("native", factory)]),
        model_factory=lambda: model, checkpointer=saver, model_ownership=ownership)


def tool_message(name="write_record", arguments=None):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments or {"value": "ok"},
                                              "id": "native-call", "type": "tool_call"}])


class AsyncRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_tool_model_saver_keep_loop_and_support_backpressure(self):
        loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), asyncio.Event()
        called = []

        @tool
        async def write_record(value: str) -> str:
            """Write a test-owned record."""
            self.assertIs(asyncio.get_running_loop(), loop)
            entered.set()
            await release.wait()
            called.append(value)
            return value

        async def authorize(**kwargs):
            self.assertIs(asyncio.get_running_loop(), loop)
            return AuthorizationDecision(kwargs["arguments"] == {"value": "ok"})

        saver = AsyncOnlySaver()
        model = AsyncModel([tool_message(), AIMessage(content="done")])
        service = native_service(model, write_record, saver, authorizer=authorize)
        async def collect():
            return [event async for event in service.astream(RunRequest("async", "native", "write"),
                                                             RunContext(CancellationToken()))]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 2)
        await asyncio.sleep(0)  # The active tool cannot block this host task.
        self.assertFalse(task.done())
        release.set()
        events = await asyncio.wait_for(task, 2)
        self.assertEqual(called, ["ok"])
        self.assertEqual(events[-1].payload["result"].output, "done")
        self.assertEqual(model.closed, 1)
        self.assertTrue(saver.loops and all(item is loop for item in saver.loops + model.loops))

    async def test_unknown_native_tool_is_denied_before_io(self):
        called = []
        @tool
        async def write_record(value: str) -> str:
            """Write a test-owned record."""
            called.append(value)
            return value
        model = AsyncModel([tool_message(), AIMessage(content="denied")])
        events = [event async for event in native_service(model, write_record, None).astream(
            RunRequest("denied", "native", "write"), RunContext(CancellationToken()))]
        self.assertEqual(called, [])
        self.assertTrue(any(event.kind == "tool.completed" and event.payload.get("errorCode") == "tool_not_allowed"
                            for event in events))

    async def test_hitl_resume_binds_checkpoint_interrupt_and_flow_fingerprint(self):
        called = []
        @tool
        async def write_record(value: str) -> str:
            """Write after a host approval."""
            called.append(value)
            return value
        saver = AsyncOnlySaver()
        model = AsyncModel([tool_message(), AIMessage(content="approved")])
        service = native_service(model, write_record, saver,
            authorizer=lambda **_: AuthorizationDecision(True), ownership="borrowed",
            middleware=[HumanInTheLoopMiddleware(interrupt_on={"write_record": True})])
        request = RunRequest("review", "native", "write", binding_fingerprint="published-flow-v1")
        events = [event async for event in service.astream(request, RunContext(CancellationToken()))]
        pause = events[-1]
        self.assertEqual(pause.kind, "run.suspended")
        self.assertEqual(called, [])
        checkpoint = pause.payload["checkpoint"]["checkpoint_id"]
        interrupt_id = pause.payload["interruptions"][0]["id"]
        decision = ResumeDecision(checkpoint, {interrupt_id: {"decisions": [{"type": "approve"}]}})
        for invalid in (
            replace(request, resume=True, resume_decision=decision, binding_fingerprint="changed"),
            replace(request, resume=True, resume_decision=ResumeDecision("stale", decision.decisions)),
            replace(request, resume=True, resume_decision=ResumeDecision(checkpoint, {"other": True})),
        ):
            invalid_events = []
            with self.assertRaises(AgentContextError):
                async for event in service.astream(invalid, RunContext(CancellationToken())):
                    invalid_events.append(event)
            self.assertFalse(any(event.kind == "run.resumed" for event in invalid_events))
            self.assertEqual(called, [])
        resumed = [event async for event in service.astream(replace(request, resume=True, resume_decision=decision),
                                                            RunContext(CancellationToken()))]
        self.assertEqual(resumed[-1].kind, "run.completed")
        self.assertEqual(resumed[0].kind, "run.resumed")
        self.assertEqual(resumed[0].payload["checkpoint"]["checkpoint_id"], checkpoint)
        self.assertEqual(called, ["ok"])
        self.assertEqual(model.closed, 0)
        with self.assertRaises(AgentContextError):
            _ = [event async for event in service.astream(replace(request, resume=True, resume_decision=decision),
                                                          RunContext(CancellationToken()))]
        self.assertEqual(called, ["ok"])

    async def test_cancellation_reaches_waiting_native_tool_and_closes_owned_model(self):
        entered, closed = asyncio.Event(), asyncio.Event()
        @tool
        async def write_record(value: str) -> str:
            """Wait for cancellation without blocking the host loop."""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            return value
        token = CancellationToken()
        model = AsyncModel([tool_message()])
        service = native_service(model, write_record, None, authorizer=lambda **_: AuthorizationDecision(True))
        async def collect():
            return [event async for event in service.astream(RunRequest("cancel", "native", "wait"), RunContext(token))]
        task = asyncio.create_task(collect())
        await asyncio.wait_for(entered.wait(), 2)
        token.cancel()
        done, _ = await asyncio.wait([task], timeout=1)
        self.assertIn(task, done, "token must cancel without wait_for injecting cancellation")
        with self.assertRaises(AgentCancelled):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(model.closed, 1)

    async def test_memory_uses_async_host_port_and_async_store(self):
        loop = asyncio.get_running_loop()
        store = AsyncOnlyStore()
        class Services:
            def snapshot(self, scope):
                raise AssertionError("sync memory access")
            async def asnapshot(self, scope):
                self_loop = asyncio.get_running_loop()
                if self_loop is not loop:
                    raise AssertionError("memory changed loops")
                return MemorySnapshot(MemoryRelease("1", ("memory",), ("skills",)), None)
        binding = MemoryBinding(MemoryScope("tenant", "user", None, "session", "task", "memory"), store, Services())
        await store.aput(("memory",), "/AGENTS.md", {"content": ["Remember the supplied scope"],
                                                   "created_at": "2026-09-29", "modified_at": "2026-09-29"})
        components = build_memory_components(binding)
        await binding.arefresh()
        decision = await components.authorize_tool.__async__("read_file", {"file_path": "/memory/user/AGENTS.md"})
        self.assertTrue(decision.allowed)
        # Official loader and file backend must use the async Store methods.
        runtime = SimpleNamespace(context=SimpleNamespace(memory=binding), state={"files": {}}, store=store)
        loader = components.middleware[2]
        loader.sources = ["/memory/user/AGENTS.md"]
        loaded = await loader.abefore_agent({}, runtime, {})
        self.assertIn("Remember the supplied scope", loaded["memory_contents"]["/memory/user/AGENTS.md"])

    async def test_async_store_file_operations_preserve_upstream_semantics(self):
        from agent_core.memory.stores.async_backend import AsyncStoreBackend
        backend = AsyncStoreBackend(store=AsyncOnlyStore(), namespace=lambda _runtime: ("drafts",))
        await backend.aupload_files([("/one.md", b"needle\ntext"), ("/two.bin", b"\xff")])
        self.assertEqual((await backend.adownload_files(["/two.bin"]))[0].content, b"\xff")
        self.assertEqual((await backend.adownload_files(["/missing"]))[0].error, "file_not_found")
        self.assertEqual(len((await backend.als("/")).entries), 2)
        self.assertEqual(len((await backend.aglob("*.md", "/")).matches), 1)
        self.assertTrue((await backend.agrep("needle", "/")).matches)
        self.assertIsNone((await backend.aedit("/one.md", "needle", "updated")).error)
        self.assertIn("updated", str(await backend.aread("/one.md")))
        self.assertIsNone((await backend.adelete("/one.md")).error)

    async def test_owned_resource_cleanup_keeps_original_failure_and_event_loop(self):
        loop = asyncio.get_running_loop()
        calls = []
        model = AsyncModel([])
        @asynccontextmanager
        async def resources():
            self.assertIs(asyncio.get_running_loop(), loop)
            calls.append("open")
            try:
                yield PersistenceResources(AsyncOnlySaver(), AsyncOnlyStore())
            finally:
                self.assertIs(asyncio.get_running_loop(), loop)
                calls.append("close")
                raise RuntimeError("cleanup failure")
        async def build(request, context, resources):
            self.assertIsInstance(resources.checkpointer, AsyncOnlySaver)
            self.assertIsInstance(resources.store, AsyncOnlyStore)
            raise AgentContextError("original failure")
        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("broken", build)]),
            model_factory=lambda: model, persistence_factory=resources)
        with self.assertRaisesRegex(AgentContextError, "original failure"):
            _ = [event async for event in service.astream(RunRequest("broken", "broken", ""),
                                                        RunContext(CancellationToken()))]
        self.assertEqual(calls, ["open", "close"])
        self.assertEqual(model.closed, 1)

    async def test_tool_body_interrupt_is_not_completed_or_replayed_after_approval(self):
        from langgraph.types import interrupt
        effects = []
        @tool
        async def write_record(value: str) -> str:
            """Require approval immediately before the external write."""
            approved = interrupt({"operation": "write_record", "value": value})
            if approved:
                effects.append(value)
            return "approved" if approved else "declined"
        saver = AsyncOnlySaver()
        model = AsyncModel([tool_message(), AIMessage(content="done")])
        service = native_service(model, write_record, saver,
            authorizer=lambda **_: AuthorizationDecision(True), ownership="borrowed")
        request = RunRequest("inner-review", "native", "write")
        events = [event async for event in service.astream(request, RunContext(CancellationToken()))]
        self.assertEqual(events[-1].kind, "run.suspended")
        self.assertFalse(any(event.kind == "tool.completed" for event in events))
        self.assertEqual(effects, [])
        payload = events[-1].payload
        decision = ResumeDecision(payload["checkpoint"]["checkpoint_id"],
                                  {payload["interruptions"][0]["id"]: True})
        events = [event async for event in service.astream(replace(request, resume=True, resume_decision=decision),
                                                          RunContext(CancellationToken()))]
        self.assertEqual(events[-1].kind, "run.completed")
        self.assertEqual(effects, ["ok"])
        self.assertEqual(sum(event.kind == "tool.completed" for event in events), 1)

    async def test_sync_facade_cancels_native_wait_and_releases_on_its_single_loop(self):
        from threading import Event
        entered, closed = Event(), Event()
        loops = []
        @tool
        async def write_record(value: str) -> str:
            """Wait until the synchronous host cancels this Run."""
            loops.append(asyncio.get_running_loop())
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()
            return value
        token = CancellationToken()
        model = AsyncModel([tool_message()])
        service = native_service(model, write_record, AsyncOnlySaver(),
                                 authorizer=lambda **_: AuthorizationDecision(True))
        task = asyncio.create_task(asyncio.to_thread(lambda: list(service.stream(
            RunRequest("sync-cancel", "native", "wait"), RunContext(token)))))
        self.assertTrue(await asyncio.to_thread(entered.wait, 2))
        token.cancel()
        done, _ = await asyncio.wait([task], timeout=1)
        self.assertIn(task, done, "token must cancel without wait_for injecting cancellation")
        with self.assertRaises(AgentCancelled):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(model.closed, 1)
        self.assertEqual(len(set([*loops, *model.loops])), 1)
        self.assertIsNot(loops[0], asyncio.get_running_loop())

    async def test_model_budget_failure_retains_public_limit_error(self):
        from agent_core.contracts.errors import AgentLimitReachedError
        async def build(request, context, resources):
            raise AgentLimitReachedError("Model call budget exhausted")
        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("budget", build)]))
        with self.assertRaisesRegex(AgentLimitReachedError, "budget exhausted"):
            _ = [event async for event in service.astream(RunRequest("budget", "budget", ""),
                                                        RunContext(CancellationToken()))]

    async def test_artifact_events_project_only_explicit_workflow_products(self):
        from agent_core import Artifact, RunResult
        class OutputState(TypedDict):
            final_answer: str
            private: str
        products = (
            Artifact("file", {"uri": "artifact:one"}, {"media_type": "text/plain"}),
            Artifact("table", [{"value": 2}], {"rows": 1}),
        )
        def factory(request, context, resources):
            graph = StateGraph(OutputState)
            graph.add_node("answer", lambda _: {"final_answer": "done", "private": "never an artifact"})
            graph.add_edge(START, "answer")
            graph.add_edge("answer", END)
            return WorkflowExecution(graph.compile(), {},
                result_factory=lambda value: RunResult(value["final_answer"], products))
        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("products", factory)]))
        events = [event async for event in service.astream(RunRequest("products", "products", ""),
                                                          RunContext(CancellationToken()))]
        self.assertEqual([event.kind for event in events], ["artifact.ready", "artifact.ready", "run.completed"])
        self.assertEqual([event.sequence for event in events], [1, 2, 3])
        for index, event in enumerate(events[:-1]):
            self.assertEqual(event.payload, {"index": index, "artifact": {
                "kind": products[index].kind, "value": products[index].value, "metadata": products[index].metadata}})
            self.assertNotIn("private", event.payload)
        self.assertEqual(events[-1].payload["result"].artifacts, products)

    async def test_existing_evidence_result_remains_owned_by_host_reference_projection(self):
        from agent_core import RunResult
        from agent_core_tests.test_runtime import GenericEvidence
        evidence = GenericEvidence(reference_id="reference-one")
        class OutputState(TypedDict):
            final_answer: str
        def factory(request, context, resources):
            graph = StateGraph(OutputState)
            graph.add_node("answer", lambda _: {"final_answer": "done"})
            graph.add_edge(START, "answer")
            graph.add_edge("answer", END)
            return WorkflowExecution(graph.compile(), {},
                result_factory=lambda value: RunResult(value["final_answer"], (evidence,)))
        service = AgentService(workflows=WorkflowRegistry([WorkflowDefinition("evidence", factory)]))
        events = [event async for event in service.astream(RunRequest("evidence", "evidence", ""),
                                                          RunContext(CancellationToken()))]
        self.assertEqual([event.kind for event in events], ["run.completed"])
        self.assertEqual(events[0].payload["result"].artifacts, (evidence,))
