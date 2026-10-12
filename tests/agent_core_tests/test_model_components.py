"""Core model/context behavior without host modules or provider requests."""
from __future__ import annotations

import asyncio
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage

from agent_core.context import build_messages
from agent_core.contracts.errors import (
    AgentCancelled,
    AgentUpstreamError,
    ModelConfigurationError,
)
from agent_core.memory import HistoryMessage, SnapshotHistory
from agent_core.models import ModelProfile, build_chat_model
from agent_core.models.streaming import stream_chat_message
from agent_core.observability import InMemoryUsageObserver, record_model_call
from agent_core.runtime.cancellation import CancellationToken
from agent_core.runtime.streaming import AsyncStreamRunner


class ProviderOptionsTest(unittest.TestCase):
    def test_glm53_uses_supported_bounded_thinking_without_increasing_output_limit(self):
        from agent_core.models.providers.openai_compatible import chat_options
        for name in ("glm-5.3", "GLM-5.3-Flash"):
            with self.subTest(model=name):
                options = chat_options(ModelProfile(name, "https://open.bigmodel.cn/api/paas/v4", "test", max_tokens=4096))
                self.assertEqual(options["reasoning_effort"], "low")
                self.assertEqual(options["extra_body"], {
                    "thinking": {"type": "enabled", "clear_thinking": True}, "tool_stream": True, "max_tokens": 4096})

    def test_previous_glm_and_unrelated_gateway_keep_their_request_contracts(self):
        from agent_core.models.providers.openai_compatible import chat_options
        previous = chat_options(ModelProfile("glm-5.2", "https://open.bigmodel.cn/api/paas/v4", "test"))
        self.assertEqual(previous["extra_body"]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", previous)
        other = chat_options(ModelProfile("glm-5.3-flash", "https://example.test/v1", "test", max_tokens=512))
        self.assertNotIn("extra_body", other)
        self.assertNotIn("reasoning_effort", other)
        self.assertEqual(other["max_completion_tokens"], 512)


class ContextHistoryTest(unittest.TestCase):
    def test_snapshot_keeps_latest_six_pairs_in_chronological_order(self) -> None:
        pairs = [HistoryMessage(f"user-{i}", f"answer-{i}") for i in range(8)]
        history = SnapshotHistory(pairs)
        pairs.clear()
        messages = build_messages("current", history)
        self.assertEqual(len(history.read()), 8)
        self.assertEqual(
            [message.content for message in messages],
            [value for i in range(2, 8) for value in (f"user-{i}", f"answer-{i}")]
            + ["current"],
        )
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)

    def test_character_budget_keeps_whole_pairs_at_the_exact_boundary(self) -> None:
        pairs = [
            HistoryMessage("old", "old"),
            HistoryMessage("a" * 4_000, "b" * 4_000),
            HistoryMessage("c" * 4_000, "d" * 4_000),
        ]
        messages = build_messages("new" * 6_000, pairs)
        self.assertEqual(len(messages), 5)
        self.assertEqual(messages[0].content, "a" * 4_000)
        self.assertEqual(messages[-1].content, "new" * 6_000)

    def test_oversized_latest_pair_stops_selection_instead_of_using_stale_history(self) -> None:
        messages = build_messages(
            "current",
            [HistoryMessage("old", "answer"), HistoryMessage("x" * 16_000, "y")],
        )
        self.assertEqual([item.content for item in messages], ["current"])


class StreamAndUsageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = AsyncStreamRunner()
        self.addCleanup(self.runner.close)
        self.cancellation = CancellationToken()
        self.observer = InMemoryUsageObserver()

    def record(self, operation, *, observer=None):
        return record_model_call(
            operation,
            observer=self.observer if observer is None else observer,
            run_id="run-1",
            attempt=2,
            call_id="call-3",
            phase="decision",
        )

    def test_stream_records_only_final_merged_usage_and_preserves_tool_chunks(self) -> None:
        class Model:
            def stream(self, messages):
                yield AIMessageChunk(
                    content="hello ",
                    tool_call_chunks=[
                        {"name": "lookup", "args": '{"value":', "id": "tool-1", "index": 0}
                    ],
                )
                yield AIMessageChunk(
                    content="world",
                    tool_call_chunks=[
                        {"name": None, "args": "7}", "id": None, "index": 0}
                    ],
                )
                yield AIMessageChunk(
                    content="",
                    usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
                )

        deltas = []
        message = self.record(
            lambda: stream_chat_message(
                Model(),
                [HumanMessage("request")],
                self.cancellation,
                async_stream_runner=self.runner,
                on_text=deltas.append,
            )
        )
        self.assertEqual(message.content, "hello world")
        self.assertEqual(deltas, ["hello ", "world"])
        self.assertEqual(message.tool_calls[0]["args"], {"value": 7})
        self.assertEqual(len(self.observer.records), 1)
        usage = self.observer.records[0]
        self.assertEqual((usage.run_id, usage.attempt, usage.call_id, usage.phase),
                         ("run-1", 2, "call-3", "decision"))
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.total_tokens), (7, 3, 10))
        self.assertEqual((usage.status, usage.source), ("succeeded", "reported"))

    def test_missing_usage_is_unknown_and_reported_zero_remains_zero(self) -> None:
        first = AIMessage(content="no usage")
        self.assertIs(self.record(lambda: first), first)
        self.assertEqual(self.observer.records[0].source, "unknown")
        self.assertIsNone(self.observer.records[0].total_tokens)
        self.record(lambda: AIMessage(content="zero", usage_metadata={
            "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
        }))
        self.assertEqual(self.observer.records[1].source, "reported")
        self.assertEqual(self.observer.records[1].total_tokens, 0)

    def test_raw_provider_usage_is_read_once_when_standard_usage_is_absent(self) -> None:
        self.record(lambda: AIMessage(content="done", response_metadata={
            "token_usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        }))
        usage = self.observer.records[0]
        self.assertEqual((usage.input_tokens, usage.output_tokens, usage.total_tokens), (4, 2, 6))

    def test_failed_and_cancelled_calls_keep_original_error_and_report_unknown_once(self) -> None:
        for error, status in [(ValueError("private provider body"), "failed"),
                              (AgentCancelled("cancelled"), "cancelled")]:
            with self.subTest(status=status):
                def fail():
                    raise error
                before = len(self.observer.records)
                with self.assertRaises(type(error)) as raised:
                    self.record(fail)
                self.assertIs(raised.exception, error)
                self.assertEqual(len(self.observer.records), before + 1)
                usage = self.observer.records[-1]
                self.assertEqual((usage.status, usage.source), (status, "unknown"))
                self.assertIsNone(usage.total_tokens)
                self.assertNotIn("private provider", repr(usage))

    def test_observer_failure_cannot_change_result_or_mask_model_error(self) -> None:
        class BrokenObserver:
            def record(self, usage):
                raise RuntimeError("telemetry unavailable")

        expected = AIMessage(content="result")
        self.assertIs(self.record(lambda: expected, observer=BrokenObserver()), expected)
        error = ValueError("original")
        def fail():
            raise error
        with self.assertRaises(ValueError) as raised:
            self.record(fail, observer=BrokenObserver())
        self.assertIs(raised.exception, error)

    def test_async_streams_reuse_loop_and_close_provider_streams(self) -> None:
        loops, closed = [], []
        class Model:
            async def astream(self, messages):
                loops.append(id(asyncio.get_running_loop()))
                try:
                    yield AIMessageChunk(content="answer")
                finally:
                    closed.append(True)

        for _ in range(2):
            stream_chat_message(Model(), [], self.cancellation,
                                async_stream_runner=self.runner)
        self.assertEqual(len(set(loops)), 1)
        self.assertEqual(closed, [True, True])

    def test_provider_async_cancel_is_upstream_failure(self) -> None:
        closed = []
        class Model:
            async def astream(self, messages):
                try:
                    yield AIMessageChunk(content="partial")
                    raise asyncio.CancelledError()
                finally:
                    closed.append(True)

        with self.assertRaises(AgentUpstreamError):
            self.record(lambda: stream_chat_message(
                Model(), [], self.cancellation, async_stream_runner=self.runner,
            ))
        self.assertEqual(closed, [True])
        self.assertEqual(self.observer.records[0].status, "failed")

    def test_precancelled_call_does_not_start_provider_stream(self) -> None:
        self.cancellation.cancel()
        class Model:
            def stream(self, messages):
                raise AssertionError("provider must not be entered")

        with self.assertRaises(AgentCancelled):
            self.record(lambda: stream_chat_message(
                Model(), [], self.cancellation, async_stream_runner=self.runner,
            ))
        self.assertEqual(len(self.observer.records), 1)
        self.assertEqual(self.observer.records[0].status, "cancelled")

    def test_cancel_interrupts_registered_stream_and_closes_late_binding(self) -> None:
        closes = []
        self.cancellation.bind_stream(lambda: closes.append("active"))
        self.cancellation.cancel()
        self.cancellation.cancel()
        self.cancellation.bind_stream(lambda: closes.append("late"))
        self.assertEqual(closes, ["active", "late"])


class ExplicitModelProfileTest(unittest.TestCase):
    def test_factory_uses_explicit_profile_and_owns_clients(self) -> None:
        class SyncClient:
            def __init__(self, **options):
                self.options, self.closed = options, False
            def close(self):
                self.closed = True
        class AsyncClient(SyncClient):
            async def aclose(self):
                self.closed = True

        fake_httpx = ModuleType("httpx")
        fake_httpx.Client, fake_httpx.AsyncClient = SyncClient, AsyncClient
        fake_provider = ModuleType("langchain_openai")
        fake_provider.ChatOpenAI = lambda **options: options
        profile = ModelProfile("example-model", "https://model.example/v1", "explicit-key",
                               max_tokens=256)
        self.assertNotIn("explicit-key", repr(profile))
        with patch.dict(sys.modules, {"httpx": fake_httpx, "langchain_openai": fake_provider}):
            model = build_chat_model(profile)
        self.assertEqual(model.chat_model["model"], "example-model")
        self.assertEqual(model.chat_model["api_key"], "explicit-key")
        self.assertEqual(model.chat_model["max_completion_tokens"], 256)
        self.assertNotIn("stream_usage", model.chat_model)
        asyncio.run(model.aclose())
        model.close()
        self.assertTrue(model.chat_model["http_client"].closed)
        self.assertTrue(model.chat_model["http_async_client"].closed)

    def test_missing_configuration_fails_before_constructing_provider(self) -> None:
        with self.assertRaises(ModelConfigurationError):
            build_chat_model(ModelProfile("model", "https://model.example/v1", ""))


if __name__ == "__main__":
    unittest.main()
