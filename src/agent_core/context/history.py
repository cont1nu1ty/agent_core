"""Archive safe tool observations before community result offloading."""

from collections.abc import Mapping
import hashlib
from typing import Any
from agent_core.runtime.streaming import call_async, call_port, run_sync

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from agent_core.context.budget import aarchive_messages
from agent_core.contracts.errors import AgentContextError
from agent_core.memory.contracts import HistoryEntry


class ToolHistoryMiddleware(AgentMiddleware):
    """Innermost tool hook: upstream file tools still own the actual offload."""

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        binding = getattr(request.runtime.context, "memory", None)
        if binding is None or binding.history is None:
            return await call_async(handler, request)
        # Preserve the request even if the actual tool later fails or crashes.
        await aarchive_messages(binding, request.state["messages"])
        result = await call_async(handler, request)
        update = dict(result.update) if isinstance(result, Command) and isinstance(result.update, Mapping) else None
        messages = update.get("messages", []) if update is not None else [result] if isinstance(result, ToolMessage) else []
        copied = []
        for index, message in enumerate(messages):
            if not isinstance(message, ToolMessage):
                copied.append(message)
                continue
            call_id = request.tool_call["id"]
            if message.tool_call_id != call_id:
                raise AgentContextError("Tool history cannot claim another call identity")
            # An unfinished tool can replay after the safe observation was
            # archived but before its checkpoint committed. Preserve both
            # executions: a fresh read may legitimately return different data.
            attempt = getattr(getattr(request.runtime.context, "request", None), "attempt", 1)
            identity = "tool-" + hashlib.sha256(f"{call_id}:{index}:{attempt}".encode()).hexdigest()[:32]
            source_id = f"{binding.scope.run_id}:{identity}"
            entry = HistoryEntry(source_id, binding.scope.run_id, len(request.state["messages"]) + index,
                                 "tool", str(message.text), call_id, message.name or request.tool_call["name"],
                                 {"status": message.status, "before_offload": True, "attempt": attempt})
            try:
                await call_port(binding.history, "append", binding.scope, [entry])
            except Exception as error:
                raise AgentContextError("Safe tool history archive failed before result offload") from error
            copied.append(message.model_copy(update={"id": message.id or identity,
                "additional_kwargs": {**message.additional_kwargs, "history_source_id": source_id,
                                      "history_run_id": binding.scope.run_id, "history_archived": True}}))
        if update is not None:
            update["messages"] = copied
            return Command(update=update, goto=result.goto, graph=result.graph, resume=result.resume)
        return copied[0] if copied else result

    def wrap_tool_call(self, request, handler):
        return run_sync(self.awrap_tool_call(request, handler))
