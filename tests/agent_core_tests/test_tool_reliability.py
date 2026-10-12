"""Retry reservations and physical attempts without network or real waiting."""

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import unittest
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_core.contracts.errors import AgentCancelled, AgentContextError, AgentLimitReachedError, DuplicateToolCallError, ToolQueryTimeoutError
from agent_core.runtime.budget import ExecutionBudget
from agent_core.runtime.cancellation import CancellationToken
from agent_core.tools.contracts import AuthorizationDecision, RecoverableToolError, ToolRetryPolicy, ToolSpec
from agent_core.tools.executor import ToolExecutor
from agent_core.tools.registry import ToolRegistry
from agent_core.tools.retry import RetryController


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    amount: int = Field(ge=1)


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.waits = []
        self.on_wait = None

    def monotonic(self):
        return self.seconds

    def now(self):
        return datetime(2026, 9, 21, tzinfo=timezone.utc) + timedelta(seconds=self.seconds)

    def wait(self, seconds):
        self.waits.append(seconds)
        self.seconds += seconds
        if self.on_wait is not None:
            self.on_wait()


@dataclass(frozen=True)
class Runtime:
    timer: Clock
    execution: ExecutionBudget = field(default_factory=lambda: ExecutionBudget(retries_enabled=True))
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    deadline: datetime | None = None
    run_awaitable: object = None

    def clock(self):
        return self.timer.now()

    def with_timeout(self, seconds):
        deadline = self.clock() + timedelta(seconds=seconds)
        return replace(self, deadline=min(self.deadline, deadline) if self.deadline else deadline)

    def raise_if_cancelled(self):
        self.cancellation.raise_if_cancelled()


class RetryTest(unittest.TestCase):
    def setUp(self):
        self.timer = Clock()
        self.runtime = Runtime(self.timer)
        self.calls = []

    def definition(self, handler=None, **changes):
        def transient(_error):
            return RecoverableToolError("connection_unavailable", "连接暂不可用", retryable=True, failure_kind="transient")

        def succeed(arguments, runtime):
            self.calls.append((arguments.amount, runtime.deadline))
            return {"amount": arguments.amount}

        values = dict(name="read_value", description="Read a value", input_schema=Input,
                      effect="read",
                      handler=handler or succeed, retry_policy=ToolRetryPolicy(mode="safe_read", max_extra_retries=2),
                      recoverable_error_mapper=transient)
        values.update(changes)
        return ToolSpec(**values)

    def executor(self, definition, **kwargs):
        return ToolExecutor(registry=ToolRegistry([definition]), retry_controller=RetryController(
            wait=self.timer.wait, monotonic=self.timer.monotonic, jitter=lambda low, high: 1.0), **kwargs)

    def execute(self, definition=None, **kwargs):
        return self.executor(definition or self.definition()).execute(call_id="read-1", tool_name="read_value",
            arguments=kwargs.pop("arguments", {"amount": 1}), runtime=kwargs.pop("runtime", self.runtime), **kwargs)

    def test_safe_validation_feedback_never_echoes_values_or_unknown_keys(self):
        result = self.execute(arguments={"amount": "PRIVATE-VALUE", "PRIVATE-KEY": "PRIVATE-VALUE"})
        self.assertFalse(result.succeeded)
        self.assertNotIn("PRIVATE", result.content)
        self.assertIn({"field": "amount", "code": "int_type", "constraints": {"type": "integer", "minimum": 1}},
                      result.data["details"]["errors"])
        self.assertIn({"field": "<field>", "code": "extra_forbidden"}, result.data["details"]["errors"])
        self.assertEqual(self.calls, [])
        self.assertEqual(self.runtime.execution.logical_tool_calls, 1)

    def test_nonfinite_invalid_input_still_uses_validation_feedback(self):
        result = self.execute(arguments={"amount": float("nan")})
        self.assertFalse(result.succeeded)
        self.assertEqual(result.data["errorCode"], "invalid_tool_arguments")
        self.assertNotIn("NaN", result.content)

    def test_validation_constraints_and_enum_come_only_from_trusted_schema(self):
        class BoundedInput(BaseModel):
            amount: int = Field(ge=1, le=5)
            mode: Literal["sum", "average"]
            label: str = Field(min_length=2, max_length=8)

        result = self.execute(self.definition(input_schema=BoundedInput), arguments={
            "amount": 100, "mode": "PRIVATE-MODE", "label": "PRIVATE-LABEL"})
        errors = {item["field"]: item for item in result.data["details"]["errors"]}
        self.assertEqual(errors["amount"]["constraints"], {"type": "integer", "minimum": 1, "maximum": 5})
        self.assertEqual(errors["mode"]["constraints"], {"type": "string", "enum": ["sum", "average"]})
        self.assertEqual(errors["label"]["constraints"], {"type": "string", "minLength": 2, "maxLength": 8})
        self.assertNotIn("PRIVATE", result.content)

    def test_two_retries_share_deadline_and_one_logical_call(self):
        def handler(arguments, runtime):
            self.calls.append((arguments.amount, runtime.deadline))
            if len(self.calls) < 3:
                raise ConnectionError("private provider detail")
            return "ready"

        result = self.execute(self.definition(handler))
        self.assertTrue(result.succeeded)
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len({deadline for _, deadline in self.calls}), 1)
        self.assertAlmostEqual(sum(self.timer.waits), 1.0)
        self.assertLessEqual(max(self.timer.waits), 0.05)
        self.assertEqual(self.runtime.execution.logical_tool_calls, 1)
        self.assertEqual(self.runtime.execution.retry_reserved_count, 2)
        self.assertEqual(self.runtime.execution.snapshot()["call_reservations"]["read-1"]["handler_attempts"], 3)

    def test_retryable_flag_without_transient_classification_does_not_retry(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            raise ValueError("unavailable")

        definition = self.definition(handler, recoverable_error_mapper=lambda error:
            RecoverableToolError("configuration", "配置不可用", retryable=True, failure_kind="configuration"))
        self.assertFalse(self.execute(definition).succeeded)
        self.assertEqual(self.calls, ["called"])
        self.assertEqual(self.timer.waits, [])

    def test_retry_switch_off_keeps_original_nonreservation_semantics(self):
        self.runtime.execution.retries_enabled = False
        result = self.execute()
        self.assertTrue(result.succeeded)
        self.assertEqual(self.runtime.execution.retry_allowance("read-1"), 0)
        self.assertEqual(self.runtime.execution.pending_call_ids(), ())
        self.assertEqual(self.runtime.execution.snapshot()["tool_execution_attempts_by_call"], {"read-1": 1})

    def test_exhaustion_stops_after_three_attempts(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            raise ConnectionError("unavailable")

        self.assertFalse(self.execute(self.definition(handler)).succeeded)
        self.assertEqual(len(self.calls), 3)

    def test_retry_after_too_long_or_longer_than_remaining_skips_retry(self):
        for retry_after, timeout in ((2.1, 10.0), (0.5, 0.4)):
            with self.subTest(retry_after=retry_after):
                self.setUp()

                def handler(_arguments, _runtime):
                    self.calls.append("called")
                    raise ConnectionError("unavailable")

                definition = self.definition(handler, timeout_seconds=timeout,
                    recoverable_error_mapper=lambda error: RecoverableToolError(
                        "busy", "稍后重试", retryable=True, failure_kind="transient", retry_after_seconds=retry_after))
                self.assertFalse(self.execute(definition).succeeded)
                self.assertEqual(self.calls, ["called"])
                self.assertEqual(self.timer.waits, [])

    def test_accepted_retry_after_is_honored_without_resetting_deadline(self):
        def handler(_arguments, runtime):
            self.calls.append(runtime.deadline)
            if len(self.calls) == 1:
                raise ConnectionError("busy")
            return "ready"

        definition = self.definition(handler, recoverable_error_mapper=lambda error: RecoverableToolError(
            "busy", "稍后重试", retryable=True, failure_kind="transient", retry_after_seconds=2.0))
        self.assertTrue(self.execute(definition).succeeded)
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(len(set(self.calls)), 1)
        self.assertAlmostEqual(sum(self.timer.waits), 2.0)

    def test_host_deadline_shorter_than_tool_timeout_is_shared(self):
        def handler(_arguments, runtime):
            self.calls.append(runtime.deadline)
            self.timer.seconds += 0.2
            raise ConnectionError("busy")

        deadline = self.timer.now() + timedelta(seconds=0.3)
        result = self.execute(self.definition(handler, timeout_seconds=10), runtime=replace(self.runtime, deadline=deadline))
        self.assertFalse(result.succeeded)
        self.assertEqual(self.calls, [deadline])
        self.assertEqual(self.timer.waits, [])

    def test_cancel_during_wait_never_starts_second_attempt(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            raise ConnectionError("unavailable")

        self.timer.on_wait = self.runtime.cancellation.cancel
        with self.assertRaises(AgentCancelled):
            self.execute(self.definition(handler))
        self.assertEqual(self.calls, ["called"])

    def test_permission_is_checked_again_before_retry(self):
        class Authorizer:
            calls = 0

            def authorize(self, definition, runtime):
                self.calls += 1
                return AuthorizationDecision(self.calls == 1, "permission_revoked", "权限已撤回")

        def handler(_arguments, _runtime):
            self.calls.append("called")
            raise ConnectionError("unavailable")

        executor = self.executor(self.definition(handler), authorizer=Authorizer())
        result = executor.execute(call_id="read-1", tool_name="read_value", arguments={"amount": 1}, runtime=self.runtime)
        self.assertEqual(result.data["errorCode"], "permission_revoked")
        self.assertEqual(self.calls, ["called"])

    def test_deadline_exhaustion_precedes_transient_mapping(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            self.timer.seconds += 1.0
            raise ConnectionError("unavailable")

        with self.assertRaises(ToolQueryTimeoutError):
            self.execute(self.definition(handler, timeout_seconds=1.0))
        self.assertEqual(self.calls, ["called"])
        self.assertEqual(self.timer.waits, [])

    def test_late_handler_success_is_rejected_without_committing_observation(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            self.timer.seconds += 1.1
            return "too late"

        with self.assertRaises(ToolQueryTimeoutError):
            self.execute(self.definition(handler, timeout_seconds=1.0))
        self.assertEqual(self.calls, ["called"])
        self.assertEqual(self.runtime.execution.observation_bytes, 0)

    def test_raw_timeout_is_not_retried_even_if_mapper_claims_transient(self):
        def handler(_arguments, _runtime):
            self.calls.append("called")
            raise TimeoutError("deadline")

        with self.assertRaises(ToolQueryTimeoutError):
            self.execute(self.definition(handler))
        self.assertEqual(self.calls, ["called"])

    def test_result_builder_and_result_commit_are_outside_retry_loop(self):
        def invalid_result(*_args):
            raise ValueError("bad result")

        with self.assertRaises(ValueError):
            self.execute(self.definition(result_builder=invalid_result))
        self.assertEqual(len(self.calls), 1)
        self.setUp()
        self.runtime.execution.max_observation_bytes = 1
        self.assertFalse(self.execute().succeeded)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.timer.waits, [])

    def test_pre_reserved_call_is_claimed_without_double_counting(self):
        budget = self.runtime.execution
        budget.reserve_call(call_id="read-1", tool_name="read_value", max_calls_for_tool=1,
            max_extra_retries=2, arguments={"amount": 1}, tool_version="1.0.0")
        self.assertFalse(budget.is_available("read_value", 1))
        self.assertTrue(budget.is_available("read_value", 1, call_id="read-1"))
        self.assertTrue(self.execute(self.definition(max_calls_per_run=1)).succeeded)
        self.assertEqual(budget.logical_tool_calls, 1)
        self.assertEqual(budget.calls_by_tool, {"read_value": 1})

    def test_same_runtime_checkpoint_restore_keeps_its_dispatch_permit(self):
        budget = self.runtime.execution
        budget.reserve_call(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
            max_extra_retries=2, arguments={"amount": 1}, tool_version="1.0.0")
        budget.restore(budget.snapshot())
        self.assertTrue(self.execute().succeeded)
        self.assertEqual(budget.logical_tool_calls, 1)

    def test_nonretry_handler_attempts_are_observed_without_driver_claims(self):
        self.assertTrue(self.execute(self.definition(retry_policy=ToolRetryPolicy())).succeeded)
        saved = self.runtime.execution.snapshot()
        self.assertEqual(saved["tool_execution_attempts_by_call"], {"read-1": 1})
        self.assertEqual(saved["tool_execution_attempts"], 1)
        restored = ExecutionBudget()
        restored.restore(saved)
        self.assertEqual(restored.snapshot()["tool_execution_attempts_by_call"], {"read-1": 1})


class ReservationTest(unittest.TestCase):
    def reserve(self, budget, call_id="read-1", **changes):
        values = dict(call_id=call_id, tool_name="read_value", max_calls_for_tool=8,
                      max_extra_retries=2, arguments={"amount": 1}, tool_version="1.0.0")
        values.update(changes)
        return budget.reserve_call(**values)

    def test_restored_pending_call_cannot_replay_and_unknown_keeps_allowance(self):
        original = ExecutionBudget(retries_enabled=True)
        self.reserve(original)
        saved_before_io = original.snapshot()
        original.begin_attempt(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
                               arguments={"amount": 1}, tool_version="1.0.0")
        original.record_handler_attempt("read-1")
        original.record_handler_attempt("read-1")
        restored = ExecutionBudget(retries_enabled=True)
        restored.restore(saved_before_io)
        with self.assertRaises(AgentContextError):
            restored.begin_attempt(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
                                   arguments={"amount": 1}, tool_version="1.0.0")
        restored.mark_outcome_unknown("read-1")
        revision = restored.revision
        restored.mark_outcome_unknown("read-1")
        self.assertEqual(restored.revision, revision)
        self.assertEqual(restored.retry_reserved_count, 2)
        self.assertEqual(restored.logical_tool_calls, 1)
        self.assertEqual(restored.pending_call_ids(), ())
        self.assertIsNone(restored.call_reservations["read-1"]["handler_attempts"])
        self.assertEqual(restored.snapshot()["tool_execution_attempts_by_call"], {"read-1": None})
        self.assertIsNone(restored.snapshot()["tool_execution_attempts"])
        self.assertEqual(restored.snapshot()["known_tool_execution_attempts"], 0)

    def test_reservation_hash_matches_arguments_without_storing_them(self):
        budget = ExecutionBudget(retries_enabled=True)
        self.reserve(budget, arguments={"amount": "PRIVATE-VALUE"})
        self.assertNotIn("PRIVATE-VALUE", str(budget.snapshot()))
        with self.assertRaises(AgentContextError):
            budget.begin_attempt(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
                                 arguments={"amount": 2}, tool_version="1.0.0")

    def test_shared_retry_allowance_is_reserved_once_and_never_refunded(self):
        budget = ExecutionBudget(retries_enabled=True)
        allowances = [self.reserve(budget, str(index)) for index in range(5)]
        self.assertEqual(allowances, [2, 2, 2, 2, 0])
        budget.complete_attempt("0")
        self.assertEqual(budget.retry_reserved_count, 8)
        self.assertEqual(budget.snapshot()["retry_slots_reserved"], 8)
        self.assertNotIn("retry_reserved_count", budget.snapshot())

    def test_runtime_extra_retry_limit_caps_tool_policy(self):
        budget = ExecutionBudget(retries_enabled=True, max_extra_retries=1)
        self.assertEqual(self.reserve(budget), 1)

    def test_batch_order_reserves_ordinary_call_before_later_retry_call(self):
        budget = ExecutionBudget(retries_enabled=True, max_external_tool_calls=1)
        self.reserve(budget, "normal", tool_name="ordinary_read", retry_protocol=False, max_extra_retries=0)
        with self.assertRaises(AgentLimitReachedError):
            self.reserve(budget, "retry")
        self.assertEqual(budget.calls_by_tool, {"ordinary_read": 1})
        self.assertEqual(budget.retry_reserved_count, 0)
        budget.begin_attempt(call_id="normal", tool_name="ordinary_read", max_calls_for_tool=8,
                             arguments={"amount": 1}, tool_version="1.0.0")
        self.assertEqual(budget.record_handler_attempt("normal"), 1)
        self.assertEqual(budget.logical_tool_calls, 2)
        self.assertEqual(budget.used_call_ids, {"normal", "retry"})

    def test_ordinary_reservation_restores_existing_single_dispatch_replay(self):
        original = ExecutionBudget(retries_enabled=True)
        self.reserve(original, retry_protocol=False, max_extra_retries=0)
        original.begin_attempt(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
                               arguments={"amount": 1}, tool_version="1.0.0")
        original.record_handler_attempt("read-1")
        restored = ExecutionBudget(retries_enabled=True)
        restored.restore(original.snapshot())
        self.assertEqual(restored.pending_call_ids(), ())
        self.assertEqual(restored.pending_call_ids(retry_protocol_only=False), ("read-1",))
        restored.begin_attempt(call_id="read-1", tool_name="read_value", max_calls_for_tool=8,
                               arguments={"amount": 1}, tool_version="1.0.0")
        self.assertEqual(restored.record_handler_attempt("read-1"), 2)
        with self.assertRaises(AgentLimitReachedError):
            restored.record_handler_attempt("read-1")
        self.assertEqual(restored.logical_tool_calls, 1)
        self.assertEqual(restored.retry_reserved_count, 0)

    def test_tool_and_group_exhaustion_are_local(self):
        budget = ExecutionBudget(max_external_tool_calls=1, discovery_enabled=True, max_discovery_tool_calls=1)
        budget.begin_attempt(call_id="one", tool_name="read_value", max_calls_for_tool=1)
        with self.assertRaises(AgentLimitReachedError):
            budget.begin_attempt(call_id="two", tool_name="read_other", max_calls_for_tool=8)
        self.assertIsNone(budget.stop_reason)
        self.assertTrue(budget.is_available("memory_read", 8, "memory"))
        budget.begin_attempt(call_id="three", tool_name="memory_read", max_calls_for_tool=8, budget_group="memory")
        budget.begin_attempt(call_id="four", tool_name="search_tools", max_calls_for_tool=8, budget_group="discovery")
        self.assertFalse(budget.is_available("search_tools", 8, "discovery"))
        self.assertTrue(budget.is_available("memory_read", 8, "memory"))
        self.assertIsNone(budget.stop_reason)
        self.assertEqual(budget.logical_tool_calls, 4)
        self.assertEqual(budget.used_call_ids, {"one", "two", "three", "four"})
        self.assertEqual(budget.calls_by_tool, {"read_value": 1, "memory_read": 1, "search_tools": 1})
        self.assertEqual(budget.calls_by_category, {"external": 1, "memory": 1, "discovery": 1})

    def test_local_denials_consume_global_capacity_once_without_local_slots(self):
        budget = ExecutionBudget(max_tool_calls=4, max_external_tool_calls=1)
        budget.begin_attempt(call_id="accepted", tool_name="read_value", max_calls_for_tool=1)
        for identity, name in (("per-tool-denied", "read_value"), ("group-denied", "read_other")):
            with self.assertRaises(AgentLimitReachedError):
                budget.begin_attempt(call_id=identity, tool_name=name, max_calls_for_tool=1)
        self.assertEqual(budget.logical_tool_calls, 3)
        self.assertEqual(budget.calls_by_tool, {"read_value": 1})
        self.assertEqual(budget.calls_by_category, {"external": 1})
        self.assertIsNone(budget.stop_reason)
        with self.assertRaises(DuplicateToolCallError):
            budget.begin_attempt(call_id="per-tool-denied", tool_name="read_value", max_calls_for_tool=1)
        self.assertEqual(budget.logical_tool_calls, 3)
        budget.begin_attempt(call_id="memory", tool_name="memory_read", max_calls_for_tool=8, budget_group="memory")
        with self.assertRaises(AgentLimitReachedError):
            budget.begin_attempt(call_id="beyond-global", tool_name="memory_read", max_calls_for_tool=8, budget_group="memory")
        self.assertEqual(budget.logical_tool_calls, 4)
        self.assertNotIn("beyond-global", budget.used_call_ids)
        self.assertEqual(budget.stop_reason, "tool_call_limit")

    def test_tool_revocation_is_idempotent_and_disables_reserved_call_visibility(self):
        budget = ExecutionBudget(retries_enabled=True)
        self.reserve(budget)
        budget.revoke_tool("read_value")
        revision = budget.revision
        budget.revoke_tool("read_value")
        self.assertEqual(budget.revision, revision)
        self.assertTrue(budget.is_tool_revoked("read_value"))
        self.assertFalse(budget.is_available("read_value", 8))
        self.assertFalse(budget.is_available("read_value", 8, call_id="read-1"))
        self.assertTrue(budget.is_available("another_read", 8))
        self.assertEqual(budget.snapshot()["revoked_tool_names"], ["read_value"])

    def test_restore_unions_revocations_even_from_an_older_snapshot(self):
        saved = ExecutionBudget()
        saved.revoke_tool("earlier_revoked")
        live = ExecutionBudget()
        live.begin_model_request()
        live.revoke_tool("locally_revoked")
        live.begin_model_request()
        old_revision = live.revision
        live.restore(saved.snapshot())
        self.assertGreater(live.revision, old_revision)
        self.assertEqual(live.model_requests, 2)
        self.assertTrue(live.is_tool_revoked("earlier_revoked"))
        self.assertTrue(live.is_tool_revoked("locally_revoked"))

        newer_without_revocations = ExecutionBudget()
        for _ in range(live.revision + 1):
            newer_without_revocations.begin_model_request()
        live.restore(newer_without_revocations.snapshot())
        self.assertEqual(live.revoked_tool_names, {"earlier_revoked", "locally_revoked"})
        restored = ExecutionBudget()
        restored.restore(live.snapshot())
        self.assertEqual(restored.revoked_tool_names, live.revoked_tool_names)

    def test_restore_cannot_raise_original_run_limits_or_refill_retry_allowance(self):
        limits = dict(max_tool_calls=4, max_external_tool_calls=1, max_memory_tool_calls=1,
                      max_discovery_tool_calls=1, max_retry_attempts=1, max_extra_retries=1,
                      max_observation_bytes=4, max_references=1)
        original = ExecutionBudget(**limits, retries_enabled=True, discovery_enabled=True)
        self.assertEqual(self.reserve(original), 1)
        restored = ExecutionBudget(max_tool_calls=40, max_external_tool_calls=20, max_memory_tool_calls=20,
            max_discovery_tool_calls=5, max_retry_attempts=20, max_extra_retries=2,
            max_observation_bytes=100, max_references=10, retries_enabled=True, discovery_enabled=True)
        restored.restore(original.snapshot())
        self.assertEqual(restored.snapshot()["limits"], limits)
        self.assertEqual(restored.retry_reserved_count, 1)
        self.assertFalse(restored.is_available("read_other", 8))
        with self.assertRaises(AgentLimitReachedError):
            restored.begin_attempt(call_id="denied", tool_name="read_other", max_calls_for_tool=8)
        self.assertEqual(self.reserve(restored, "memory", tool_name="memory_read", budget_group="memory"), 0)
        self.assertEqual(self.reserve(restored, "discovery", tool_name="search_tools", budget_group="discovery"), 0)
        self.assertEqual(restored.logical_tool_calls, 4)
        with self.assertRaises(AgentLimitReachedError):
            self.reserve(restored, "beyond-total", tool_name="read_other", budget_group="memory")
        self.assertEqual(restored.logical_tool_calls, 4)
        self.assertEqual(restored.retry_reserved_count, 1)

    def test_restore_preserves_stricter_current_limits_and_zero_retry_configuration(self):
        original = ExecutionBudget(max_tool_calls=6, max_extra_retries=0, max_retry_attempts=0)
        original.begin_model_request()
        saved = original.snapshot()
        restored = ExecutionBudget(max_tool_calls=2, max_extra_retries=2, max_retry_attempts=8, retries_enabled=True)
        restored.restore(saved)
        self.assertEqual(restored.max_tool_calls, 2)
        self.assertEqual(restored.max_extra_retries, 0)
        self.assertEqual(restored.max_retry_attempts, 0)
        self.assertEqual(self.reserve(restored), 0)
        self.assertEqual(restored.call_reservations["read-1"]["retry_protocol"], True)

    def test_stale_restore_still_reapplies_saved_limits_after_settings_refresh(self):
        live = ExecutionBudget(max_external_tool_calls=1)
        live.begin_model_request()
        saved = live.snapshot()
        live.begin_model_request()
        live.max_external_tool_calls = 20
        revision = live.revision
        live.restore(saved)
        self.assertEqual(live.max_external_tool_calls, 1)
        self.assertEqual(live.model_requests, 2)
        self.assertGreater(live.revision, revision)

    def test_discovery_requires_switch_and_model_request_counter_restores(self):
        budget = ExecutionBudget()
        self.assertFalse(budget.is_available("search_tools", 3, "discovery"))
        budget.begin_model_request()
        budget.begin_model_request()
        restored = ExecutionBudget()
        restored.restore(budget.snapshot())
        self.assertEqual(restored.begin_model_request(), 3)

    def test_snapshot_does_not_share_mutable_reservations(self):
        budget = ExecutionBudget(retries_enabled=True)
        self.reserve(budget)
        saved = budget.snapshot()
        budget.mark_outcome_unknown("read-1")
        self.assertEqual(saved["call_reservations"]["read-1"]["outcome"], "pending")


if __name__ == "__main__":
    unittest.main()
