"""Checkpoint failure windows through actual create_agent execution and recovery."""

from dataclasses import replace
import json
from threading import Lock
import unittest

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from agent_core.config import AgentConfig
from agent_core.contracts.errors import AgentUpstreamError
from agent_core.contracts.runs import RunRequest
from agent_core.tools import ToolRetryPolicy
from agent_core_tests.test_tool_discovery_runtime import CaptureModel, call, decision, run, service, specification


class CheckpointWriteFailure(RuntimeError):
    pass


class FaultSaver(InMemorySaver):
    """Reject public Saver writes at a selected boundary, without altering state."""

    def __init__(self, call_id, stage):
        super().__init__()
        self.call_id = call_id
        self.stage = stage
        self.armed = True
        self.failed = False
        self.rejected_writes = []
        self.persisted_reservations = []
        self._fault_lock = Lock()

    def _reservation(self, values):
        if not isinstance(values, dict):
            return None
        ledger = values.get("execution_ledger", {})
        return ledger.get("call_reservations", {}).get(self.call_id) if isinstance(ledger, dict) else None

    def _contains_result(self, values):
        reservation = self._reservation(values)
        if reservation and reservation.get("outcome") == "completed":
            return True
        messages = values.get("messages", ())
        if isinstance(messages, ToolMessage):
            messages = [messages]
        return any(isinstance(message, ToolMessage) and message.tool_call_id == self.call_id for message in messages)

    def _reject(self, operation, values):
        reservation = self._reservation(values)
        trigger = (self.stage == "reservation" and operation == "put"
                   and reservation is not None and reservation.get("outcome") == "pending")
        trigger = trigger or self.stage == "result" and self._contains_result(values)
        with self._fault_lock:
            if self.armed and (trigger or self.failed):
                self.failed = True
                self.rejected_writes.append(operation)
                raise CheckpointWriteFailure(f"injected {self.stage} persistence failure")

    def put(self, config, checkpoint, metadata, new_versions):
        values = checkpoint.get("channel_values", {})
        self._reject("put", values)
        result = super().put(config, checkpoint, metadata, new_versions)
        reservation = self._reservation(values)
        if reservation is not None:
            self.persisted_reservations.append(dict(reservation))
        return result

    def put_writes(self, config, writes, task_id, task_path=""):
        writes = list(writes)
        self._reject("put_writes", dict(writes))
        return super().put_writes(config, writes, task_id, task_path)


def retry_read(executed, *, handler=None):
    return replace(specification("read_value", executed, exposure="direct", handler=handler),
                   retry_policy=ToolRetryPolicy(mode="safe_read"))


def tool_messages(model, call_id):
    return [message for message in model.calls[0][1]
            if isinstance(message, ToolMessage) and message.tool_call_id == call_id]


class ToolRecoveryFaultTest(unittest.TestCase):
    def test_failed_reservation_checkpoint_prevents_handler_io(self):
        executed = []
        saver = FaultSaver("read-1", "reservation")
        model = CaptureModel([decision(call("read_value", "read-1", text="value"))])
        contexts = []
        with self.assertRaises(AgentUpstreamError):
            run(service(model, [retry_read(executed)], saver=saver,
                        settings=AgentConfig(tool_retry_enabled=True), contexts=contexts))
        self.assertTrue(saver.failed)
        self.assertIn("put", saver.rejected_writes)
        self.assertEqual(executed, [])
        self.assertEqual(saver.persisted_reservations, [])
        reservation = contexts[-1].tool_runtime.execution.call_reservations["read-1"]
        self.assertEqual(reservation["retry_allowance"], 2)
        self.assertEqual(reservation["handler_attempts"], 0)

    def test_success_with_lost_tool_result_is_not_replayed_after_restart(self):
        executed = []
        saver = FaultSaver("read-1", "result")
        definition = retry_read(executed)
        settings = AgentConfig(tool_retry_enabled=True)
        request = RunRequest("lost-success", "discovery-test", "read")
        first = CaptureModel([decision(call("read_value", "read-1", text="returned but not saved"))])
        with self.assertRaises(AgentUpstreamError):
            run(service(first, [definition], saver=saver, settings=settings), request)
        self.assertEqual(executed, [("read_value", "returned but not saved")])
        self.assertTrue(saver.failed)
        self.assertTrue(saver.rejected_writes)
        self.assertTrue(any(item["outcome"] == "pending" for item in saver.persisted_reservations))
        self.assertFalse(any(item["outcome"] == "completed" for item in saver.persisted_reservations))

        saver.armed = False
        resumed = CaptureModel([[AIMessage(content="Previous query outcome is unknown.")]])
        contexts = []
        run(service(resumed, [definition], saver=saver, settings=settings, contexts=contexts),
            replace(request, resume=True, attempt=2))
        self.assertEqual(executed, [("read_value", "returned but not saved")])
        messages = tool_messages(resumed, "read-1")
        self.assertEqual(len(messages), 1)
        self.assertEqual(json.loads(messages[0].text)["errorCode"], "tool_execution_interrupted")
        budget = contexts[-1].tool_runtime.execution
        self.assertEqual(budget.retry_reserved_count, 2)
        self.assertEqual(budget.call_reservations["read-1"]["outcome"], "unknown")
        self.assertIsNone(budget.snapshot()["tool_execution_attempts"])
        self.assertEqual(budget.observation_bytes, 0)

    def test_zero_retry_allowance_still_uses_outcome_unknown_recovery(self):
        for limit in ("max_tool_extra_retries", "max_tool_retry_slots"):
            with self.subTest(limit=limit):
                executed = []

                def crash(args, _runtime):
                    executed.append(args.text)
                    raise RuntimeError("interrupted before returning the first result")

                definition = retry_read([], handler=crash)
                settings = replace(AgentConfig(tool_retry_enabled=True), **{limit: 0})
                saver = InMemorySaver()
                request = RunRequest(f"zero-{limit}", "discovery-test", "read")
                first = CaptureModel([decision(call("read_value", "read-1", text="once"))])
                with self.assertRaises(AgentUpstreamError):
                    run(service(first, [definition], saver=saver, settings=settings), request)
                resumed = CaptureModel([[AIMessage(content="Outcome unknown.")]])
                contexts = []
                run(service(resumed, [definition], saver=saver, settings=settings, contexts=contexts),
                    replace(request, resume=True, attempt=2))
                self.assertEqual(executed, ["once"])
                self.assertEqual(len(tool_messages(resumed, "read-1")), 1)
                budget = contexts[-1].tool_runtime.execution
                self.assertEqual(budget.retry_reserved_count, 0)
                self.assertEqual(budget.call_reservations["read-1"]["outcome"], "unknown")
                self.assertTrue(budget.call_reservations["read-1"]["retry_protocol"])
                self.assertIsNone(budget.call_reservations["read-1"]["handler_attempts"])

    def test_repeated_recovery_keeps_one_unknown_message_and_reserved_allowance(self):
        executed = []

        def crash(args, _runtime):
            executed.append(args.text)
            raise RuntimeError("initial tool process loss")

        definition = retry_read([], handler=crash)
        settings = AgentConfig(tool_retry_enabled=True)
        saver = InMemorySaver()
        request = RunRequest("repeated-recovery", "discovery-test", "read")
        first = CaptureModel([decision(call("read_value", "read-1", text="once"))])
        with self.assertRaises(AgentUpstreamError):
            run(service(first, [definition], saver=saver, settings=settings), request)

        interrupted_resume = CaptureModel([RuntimeError("model fails after recovery checkpoint")])
        first_contexts = []
        with self.assertRaises(AgentUpstreamError):
            run(service(interrupted_resume, [definition], saver=saver, settings=settings, contexts=first_contexts),
                replace(request, resume=True, attempt=2))
        first_messages = tool_messages(interrupted_resume, "read-1")
        self.assertEqual(len(first_messages), 1)
        self.assertEqual(first_contexts[-1].tool_runtime.execution.retry_reserved_count, 2)

        resumed = CaptureModel([[AIMessage(content="Outcome still unknown.")]])
        second_contexts = []
        run(service(resumed, [definition], saver=saver, settings=settings, contexts=second_contexts),
            replace(request, resume=True, attempt=3))
        messages = tool_messages(resumed, "read-1")
        self.assertEqual([message.id for message in messages], [first_messages[0].id])
        self.assertEqual(executed, ["once"])
        budget = second_contexts[-1].tool_runtime.execution
        self.assertEqual(budget.retry_reserved_count, 2)
        self.assertEqual(budget.logical_tool_calls, 1)
        self.assertEqual(budget.pending_call_ids(), ())
        self.assertIsNone(budget.snapshot()["tool_execution_attempts_by_call"]["read-1"])

        finished = CaptureModel([])
        run(service(finished, [definition], saver=saver, settings=settings), replace(request, resume=True, attempt=4))
        self.assertEqual(finished.calls, [])
        self.assertEqual(executed, ["once"])

    def test_model_batch_order_gives_last_group_slot_to_ordinary_call(self):
        executed = []
        ordinary = specification("ordinary_read", executed, exposure="direct")
        retrying = retry_read(executed)
        model = CaptureModel([
            decision(call("ordinary_read", "ordinary", text="first"), call("read_value", "retrying", text="second")),
            [AIMessage(content="Only the first permitted query completed.")],
        ])
        contexts = []
        run(service(model, [ordinary, retrying], saver=InMemorySaver(),
                    settings=AgentConfig(tool_retry_enabled=True, max_external_tool_calls=1), contexts=contexts))
        self.assertEqual(executed, [("ordinary_read", "first")])
        budget = contexts[-1].tool_runtime.execution
        self.assertEqual(budget.calls_by_category["external"], 1)
        self.assertEqual(budget.calls_by_tool, {"ordinary_read": 1})
        self.assertEqual(budget.retry_reserved_count, 0)
        self.assertEqual(budget.snapshot()["tool_execution_attempts_by_call"], {"ordinary": 1})
        self.assertFalse(budget.call_reservations["ordinary"]["retry_protocol"])
        self.assertEqual(tool_messages(model, "retrying"), [])  # No tool message existed at the initial model request.
        rejected = [message for message in model.calls[1][1] if isinstance(message, ToolMessage) and message.tool_call_id == "retrying"]
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].status, "error")


if __name__ == "__main__":
    unittest.main()
