"""Bound the history included in the next model invocation."""
from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from agent_core.memory.history import HistoryMessage, HistoryReader
from agent_core.context.compaction import compact_history


def build_messages(
    input: str,
    history: HistoryReader | Sequence[HistoryMessage],
) -> list[BaseMessage]:
    """Keep the most recent six pairs within the existing 16,000-char limit.

    The host owns eligibility (session, dataset, completed status). A pair is
    indivisible, and a pair that exceeds the remaining budget ends selection;
    older pairs must not displace the latest conversation. The new user input
    is always appended separately, matching the existing request contract.
    """
    read = getattr(history, "read", None)
    pairs = list(read() if callable(read) else history)
    # A prepared session snapshot already accounts for all older sources through
    # its derived summary. Never silently reapply the legacy six-pair filter.
    selected = pairs if getattr(history, "preselected", False) else compact_history(pairs)
    messages: list[BaseMessage] = []
    for pair in selected:
        def metadata(source_id):
            return ({"history_source_id": source_id, "history_run_id": pair.source_run_id, "history_backfill": True,
                     "context_task_id": pair.task_id, "history_archived": bool(getattr(history, "archived", False))}
                    if source_id and pair.source_run_id else {})
        messages.extend(
            [HumanMessage(pair.user_message, id=pair.user_source_id,
                          additional_kwargs=metadata(pair.user_source_id)),
             AIMessage(content=pair.assistant_answer, id=pair.assistant_source_id,
                       additional_kwargs=metadata(pair.assistant_source_id))]
        )
    messages.append(HumanMessage(input))
    return messages


async def abuild_messages(input, history):
    """Read asynchronous history before using the shared pure projection."""
    from types import SimpleNamespace
    from agent_core.runtime.streaming import call_port
    if not callable(getattr(history, "aread", None)) and not callable(getattr(history, "read", None)):
        return build_messages(input, history)
    pairs = await call_port(history, "read")
    snapshot = SimpleNamespace(read=lambda: pairs,
        preselected=getattr(history, "preselected", False), archived=getattr(history, "archived", False))
    return build_messages(input, snapshot)
