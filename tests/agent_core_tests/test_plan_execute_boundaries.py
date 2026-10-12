"""Plan boundaries through real create_agent and Rubric graphs with scripted I/O."""

import unittest

from langchain_core.messages import AIMessage

from agent_core.contracts.errors import AgentContextError, ToolPolicyDeniedError
from agent_core.planning.plan_execute import PlanActionError, PlanInput, plan_result, project_plan
from agent_core_tests import test_plan_execute as fixtures
from agent_core_tests.test_plan_execute import ScriptedModel


response, draft, preview, report, item, grade = (
    fixtures.response, fixtures.draft, fixtures.preview, fixtures.report, fixtures.item, fixtures.grade,
)


class Harness:
    """Reuse fixture setup without inheriting or rediscovering its test cases."""

    setUp = fixtures.PlanExecuteTests.setUp
    context = fixtures.PlanExecuteTests.context
    graph = fixtures.PlanExecuteTests.graph
    invoke = fixtures.PlanExecuteTests.invoke
    approved = fixtures.PlanExecuteTests.approved


def answer(text, identity):
    return [AIMessage(content=text, id=identity)]


def failed_grade(identity):
    return grade("needs_revision", identity=identity, criteria=[{
        "name": "A1: Report the observed text", "passed": False,
        "gap": "Include the actual observed text in the answer",
    }])


def replan(remaining_steps, identity, **changes):
    return response("ReplanDecision", {
        "kind": "continue", "reason": "Retry the unresolved observation using the available evidence",
        "remaining_steps": remaining_steps, **changes,
    }, identity)


class PlanExecuteBoundaryTests(Harness, unittest.TestCase):
    def test_rubric_revision_uses_feedback_and_only_completes_after_satisfied(self):
        model = ScriptedModel([
            preview(), response("echo_text", {"text": "hello"}, "echo1"), answer("ReAct observed the text", "react-answer"), report(),
            answer("The observation is available", "initial-final"), failed_grade("grade1"),
            answer("Observed text 1: hello", "revised-final"), grade(identity="grade2"),
        ])
        ready, _ = self.invoke(model)
        done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(done["plan_state"]["phase"], "completed")
        self.assertEqual(done["plan_state"]["validation"]["status"], "satisfied")
        self.assertEqual(done["plan_state"]["validation"]["iterations"], 2)
        self.assertEqual(done["final_answer"], "Observed text 1: hello")
        self.assertTrue(any("Include the actual observed text" in message.text for message in model.calls[6][1]))
        self.assertEqual(self.tool_calls, ["hello"])
        self.assertEqual(context.model_budget.calls, 8)

    def test_rubric_failed_does_not_complete_or_retry_tools(self):
        model = ScriptedModel([
            preview(), response("echo_text", {"text": "hello"}, "echo1"), answer("ReAct observed the text", "react-answer"), report(),
            answer("Observed text 1", "final"), grade("failed", criteria=[]),
        ])
        ready, _ = self.invoke(model)
        done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(done["plan_state"]["phase"], "partial")
        self.assertEqual(done["plan_state"]["validation"]["status"], "failed")
        self.assertEqual(done["plan_state"]["steps"][0]["status"], "completed")
        self.assertTrue(done["plan_state"]["unresolved"])
        self.assertIn("任务尚未完成", done["final_answer"])
        self.assertEqual(self.tool_calls, ["hello"])
        self.assertEqual(context.model_budget.calls, 6)

    def test_grader_exception_is_partial_with_unknown_usage_retained(self):
        model = ScriptedModel([
            preview(), response("echo_text", {"text": "hello"}, "echo1"), answer("ReAct observed the text", "react-answer"), report(),
            answer("Observed text 1", "final"), RuntimeError("scripted grader unavailable"),
        ])
        ready, _ = self.invoke(model)
        with self.assertLogs("deepagents.middleware.rubric", level="ERROR"):
            done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(done["plan_state"]["phase"], "partial")
        self.assertEqual(done["plan_state"]["validation"]["status"], "grader_error")
        self.assertEqual(done["plan_state"]["validation"]["iterations"], 1)
        self.assertEqual(context.model_budget.calls, 6)
        self.assertEqual(done["plan_state"]["budget"]["unknownUsageCalls"], 3)
        self.assertIsNone(done["plan_state"]["budget"]["totalTokens"])
        self.assertEqual(self.tool_calls, ["hello"])

    def test_rubric_iteration_cap_preserves_partial_result(self):
        model = ScriptedModel([
            preview(), response("echo_text", {"text": "hello"}, "echo1"), answer("ReAct observed the text", "react-answer"), report(),
            answer("The observation is available", "initial-final"), failed_grade("grade1"),
            answer("The observation still needs explanation", "revised-final"), failed_grade("grade2"),
        ])
        ready, _ = self.invoke(model, max_rubric_iterations=2)
        done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2",
                                    max_rubric_iterations=2)
        self.assertEqual(done["plan_state"]["phase"], "partial")
        self.assertEqual(done["plan_state"]["validation"]["status"], "max_iterations_reached")
        self.assertEqual(done["plan_state"]["validation"]["iterations"], 2)
        self.assertFalse(done["plan_state"]["validation"]["criteria"][0]["passed"])
        self.assertIn("Include the actual observed text", done["plan_state"]["unresolved"][0])
        self.assertEqual(context.model_budget.calls, 8)
        self.assertEqual(self.tool_calls, ["hello"])

    def test_replan_retries_failed_step_without_repeating_completed_work(self):
        retry = {"id": "step2", "goal": "Observe the corrected second text", "expected_evidence": "Tool response",
                 "depends_on": ["step1"]}
        model = ScriptedModel([
            preview(2), response("echo_text", {"text": "one"}, "echo1"),
            response("echo_text", {"text": "unreliable two"}, "echo2"),
            answer("One observed; second result unreliable", "react-first"),
            report(item(1), item(2, status="failed", result="Second observation is unreliable", unresolved=["Retry text 2"])),
            replan([retry], "replan1"), response("echo_text", {"text": "corrected two"}, "echo3"),
            answer("Corrected second observation", "react-retry"),
            report(item(2, result="Corrected second observation"), identity="report-retry"),
            answer("Observed one and corrected two", "final"), grade(),
        ])
        ready, _ = self.invoke(model)
        original = ready["plan_state"]
        done, context = self.invoke(model, self.approved(original), original, "run2")
        plan = done["plan_state"]
        self.assertEqual(plan["phase"], "completed")
        self.assertEqual(plan["revision"], original["revision"] + 1)
        self.assertEqual(plan["replans"], 1)
        self.assertEqual(plan["approved_scope"]["scope"], original["scope"])
        self.assertEqual(plan["steps"][0]["goal"], original["steps"][0]["goal"])
        self.assertEqual(plan["steps"][0]["result"], "Observed text 1")
        self.assertTrue(plan["steps"][0]["source_ids"])
        self.assertEqual(plan["steps"][1]["goal"], retry["goal"])
        self.assertEqual(plan["steps"][1]["result"], "Corrected second observation")
        self.assertEqual(self.tool_calls, ["one", "unreliable two", "corrected two"])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 3)
        history = project_plan(plan)["stepHistory"]
        self.assertEqual([item["status"] for item in history], ["completed", "failed", "completed"])
        self.assertEqual(history[0]["sourceIds"], plan["steps"][0]["source_ids"])
        self.assertEqual(history[1]["result"], "Second observation is unreliable")
        self.assertEqual(history[1]["unresolved"], ["Retry text 2"])
        self.assertTrue(history[1]["sourceIds"])
        self.assertEqual(history[1]["revision"], original["revision"])
        self.assertEqual(history[2]["revision"], plan["revision"])

    def test_automatic_replan_cannot_replace_a_completed_step(self):
        model = ScriptedModel([
            preview(2), response("echo_text", {"text": "one"}, "echo1"),
            answer("First done, second needs a different query", "react-first"),
            report(item(1), item(2, status="blocked", unresolved=["Second observation needs a different query"]),
                   new_conditions=["Second observation needs a different query"]),
            replan(draft(2)["steps"], "rewrite-completed"),
        ])
        ready, _ = self.invoke(model)
        with self.assertRaisesRegex(AgentContextError, "cannot replace completed"):
            self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2")
        self.assertEqual(self.tool_calls, ["one"])
        self.assertEqual(len(model.calls), 5)

    def test_changed_scope_returns_ready_and_requires_new_confirmation(self):
        model = ScriptedModel([
            preview(), response("echo_text", {"text": "one"}, "echo1"),
            answer("Need permission to observe a second text", "react-first"),
            report(item(status="blocked", unresolved=["Need permission to observe a second text"])),
            replan([], "scope-change", kind="confirm_scope", proposed_plan=draft(2, scope=["Echo two inputs"])),
        ])
        ready, _ = self.invoke(model)
        old_action = self.approved(ready["plan_state"])
        changed, context = self.invoke(model, old_action, ready["plan_state"], "run2")
        plan = changed["plan_state"]
        self.assertEqual(plan["phase"], "ready")
        self.assertEqual(plan["scope"], ["Echo two inputs"])
        self.assertIsNone(plan["approved_scope"])
        self.assertIsNone(plan["approved_revision"])
        self.assertTrue(all(item["status"] == "pending" for item in plan["steps"]))
        with self.assertRaises(PlanActionError):
            self.invoke(model, old_action, plan, "run3")
        self.assertEqual(context.model_budget.calls, 5)
        self.assertEqual(len(model.calls), 5)
        self.assertEqual(self.tool_calls, ["one"])

    def test_supplement_revises_preview_and_invalidates_old_execute_action(self):
        model = ScriptedModel([preview(), preview(identity="supplement", constraints=["Read only", "Use exact text"])])
        ready, _ = self.invoke(model)
        old_action = self.approved(ready["plan_state"])
        action = PlanInput(kind="message", plan_id=old_action.plan_id, expected_revision=old_action.expected_revision)
        revised, context = self.invoke(model, action, ready["plan_state"], "run2")
        self.assertEqual(revised["plan_state"]["phase"], "ready")
        self.assertGreater(revised["plan_state"]["revision"], old_action.expected_revision)
        self.assertIn("Use exact text", revised["plan_state"]["constraints"])
        self.assertIsNone(revised["plan_state"]["approved_revision"])
        with self.assertRaises(PlanActionError):
            self.invoke(model, old_action, revised["plan_state"], "run3")
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(context.model_budget.calls, 2)
        self.assertEqual(self.tool_calls, [])

    def test_three_replans_are_a_hard_limit_and_fourth_attempt_stays_partial(self):
        scripted = [preview()]
        for attempt in range(4):
            scripted.extend([
                response("echo_text", {"text": f"attempt {attempt}"}, f"echo{attempt}"),
                answer(f"Unreliable attempt {attempt}", f"react{attempt}"),
                report(item(status="failed", result=f"Unreliable attempt {attempt}",
                    unresolved=["Observation remains unreliable"]), identity=f"result{attempt}"),
            ])
            if attempt < 3:
                scripted.append(replan(draft()["steps"], f"replan{attempt}"))
        model = ScriptedModel(scripted)
        ready, _ = self.invoke(model, max_replans=3)
        done, context = self.invoke(model, self.approved(ready["plan_state"]), ready["plan_state"], "run2", max_replans=3)
        self.assertEqual(done["plan_state"]["phase"], "partial")
        self.assertEqual(done["plan_state"]["replans"], 3)
        self.assertEqual(done["plan_state"]["budget"]["replans"], 3)
        self.assertEqual(done["plan_state"]["budget"]["maxReplans"], 3)
        self.assertIn("达到剩余计划调整次数上限", done["plan_state"]["unresolved"])
        self.assertEqual(len(model.calls), 16)
        self.assertEqual(context.model_budget.calls, 16)
        self.assertEqual(self.tool_calls, [f"attempt {attempt}" for attempt in range(4)])

    def test_model_allowance_counts_supplements_and_is_not_reset_by_new_runs(self):
        model = ScriptedModel([
            preview(), preview(identity="supplement"), response("echo_text", {"text": "hello"}, "echo1"),
        ])
        ready, _ = self.invoke(model, max_model_calls=3)
        initial = ready["plan_state"]
        action = PlanInput(kind="message", plan_id=initial["plan_id"], expected_revision=initial["revision"])
        revised, _ = self.invoke(model, action, initial, "run2", max_model_calls=3)
        exhausted, context = self.invoke(model, self.approved(revised["plan_state"]), revised["plan_state"], "run3",
                                         max_model_calls=3)
        self.assertEqual(exhausted["plan_state"]["phase"], "failed")
        self.assertEqual(exhausted["plan_state"]["budget"]["modelCalls"], 3)
        self.assertEqual(exhausted["plan_state"]["budget"]["maxModelCalls"], 3)
        self.assertEqual(context.model_budget.calls, 3)
        self.assertEqual(len(plan_result(exhausted).artifacts), 1)
        self.assertEqual(plan_result(exhausted).artifacts[0].reference_id, "R1")
        self.assertGreater(exhausted["observation_bytes"], 0)
        plan = exhausted["plan_state"]
        continuation = PlanInput(kind="message", plan_id=plan["plan_id"], expected_revision=plan["revision"])
        still_exhausted, next_context = self.invoke(model, continuation, plan, "run4", max_model_calls=3)
        self.assertEqual(still_exhausted["plan_state"]["phase"], "failed")
        self.assertEqual(next_context.model_budget.calls, 3)
        self.assertEqual(next_context.tool_runtime.execution.tool_call_count, 1)
        self.assertEqual(len(model.calls), 3)
        self.assertEqual(self.tool_calls, ["hello"])

    def test_exploration_does_not_execute_unapproved_domain_tools(self):
        model = ScriptedModel([response("echo_text", {"text": "unapproved exploration"}, "echo1")])
        with self.assertRaises(ToolPolicyDeniedError):
            self.invoke(model)
        self.assertEqual(self.tool_calls, [])
        self.assertEqual(len(model.calls), 1)

    def test_explicit_readonly_exploration_counts_tool_and_model_usage_without_executing_steps(self):
        model = ScriptedModel([
            response("echo_text", {"text": "readonly exploration"}, "explore1"), preview(),
        ])
        ready, context = self.invoke(model, exploration_tools=self.tools)
        plan = ready["plan_state"]
        self.assertEqual(plan["phase"], "ready")
        self.assertEqual(plan["steps"][0]["status"], "pending")
        self.assertIsNone(plan["approved_revision"])
        self.assertEqual(self.tool_calls, ["readonly exploration"])
        self.assertEqual(context.tool_runtime.execution.tool_call_count, 1)
        self.assertEqual(context.model_budget.calls, 2)
        self.assertEqual(plan["budget"]["modelCalls"], 2)

    def test_invalid_question_sets_are_retried_inside_bounded_structured_agent(self):
        question = {"id": "target", "title": "Which text?", "options": [{"label": "hello"}]}
        model = ScriptedModel([
            response("ExplorationResult", {"kind": "questions", "questions": [question, question]}, "duplicate"),
            response("ExplorationResult", {"kind": "questions", "questions": [
                {**question, "id": f"target{index}"} for index in range(4)
            ]}, "too-many"),
            response("ExplorationResult", {"kind": "questions", "questions": [question]}, "valid"),
        ])
        waiting, context = self.invoke(model)
        plan = waiting["plan_state"]
        self.assertEqual(plan["phase"], "clarifying")
        self.assertEqual([item["id"] for item in plan["questions"]], ["target"])
        self.assertEqual(context.model_budget.calls, 3)
        self.assertTrue(any("Question IDs must be unique" in message.text for message in model.calls[1][1]))
        self.assertTrue(any("at most 3" in message.text for message in model.calls[2][1]))
        self.assertEqual(self.tool_calls, [])


if __name__ == "__main__":
    unittest.main()
