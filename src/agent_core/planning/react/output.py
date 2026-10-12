"""Publish only completed answer candidates through application output guards."""

from collections.abc import Sequence
from typing import Any

from agent_core.contracts.errors import AgentUpstreamError, OutputValidationError
from agent_core.contracts.events import activity_label, activity_scope
from agent_core.planning.react.policy import ReActPolicy


def emit_step(runtime: Any, step_id: str, name: str, status: str, *, state: Any = None, **payload: Any) -> None:
    runtime.stream_writer({"kind": "step.state", "payload": {
        "stepId": step_id, "name": name, "status": status, "activityScopeId": activity_scope(state),
        "activityLabel": activity_label(state), **payload,
    }})


def publish_answer(
    state: dict[str, Any], runtime: Any, policy: ReActPolicy, chunks: Sequence[str],
) -> str:
    context = runtime.context
    context.cancellation.raise_if_cancelled()
    emit_step(runtime, "final_answer", policy.text("final_step"), "running", state=state)
    references = state.get("references", [])
    citation_guard = policy.citation_guard(references)
    sensitive_values = context.tool_runtime.execution.sensitive_values_snapshot()
    sensitive_guard = policy.sensitive_guard(sensitive_values)
    public_parts: list[str] = []
    length = 0

    def emit_text(text: str) -> None:
        if not text:
            return
        public_parts.append(text)
        for offset in range(0, len(text), 2048):
            context.cancellation.raise_if_cancelled()
            runtime.stream_writer({"kind": "answer.delta", "stepId": "final_answer", "payload": {
                "delta": text[offset:offset + 2048], "attempt": context.request.attempt,
            }})

    def feed(text: str) -> None:
        nonlocal length
        context.cancellation.raise_if_cancelled()
        length += len(text)
        if length > context.settings.max_answer_characters:
            raise AgentUpstreamError(policy.text("answer_too_long"))
        for safe_text in citation_guard.feed(sensitive_guard.feed(text)):
            emit_text(safe_text)

    notice = policy.limit_notice(state.get("stop_reason"))
    if notice:
        feed(notice + "\n\n")
    if not any(chunks):
        raise AgentUpstreamError(policy.text("empty_model_result"))
    for text in chunks:
        feed(text)
    trailing = sensitive_guard.finish()
    if trailing:
        for safe_text in citation_guard.feed(trailing):
            emit_text(safe_text)
    for safe_text in citation_guard.finish():
        emit_text(safe_text)
    if not references and not "".join(public_parts).strip():
        emit_text(policy.text("empty_public_answer"))
    answer = "".join(public_parts)
    if any(value in answer for value in sensitive_values):
        raise OutputValidationError(policy.text("sensitive_output"))
    emit_step(runtime, "final_answer", policy.text("final_step"), "succeeded", state=state, outputCharacters=len(answer))
    emit_step(runtime, "grounding_validation", policy.text("validation_step"), "running", state=state)
    policy.validate_output(answer, references)
    validated = {**state, "final_answer": answer}
    emit_step(runtime, "grounding_validation", policy.text("validation_step"), "succeeded", state=state,
              **policy.validation_payload(validated))
    for event in policy.after_validation_events(validated):
        runtime.stream_writer(event)
    return answer
