"""Assemble LangChain's ReAct agent using only public extension points."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool

from agent_core.contracts.errors import ToolCallingUnsupportedError
from agent_core.context.history import ToolHistoryMiddleware
from agent_core.models.langchain import StreamingChatModelAdapter
from agent_core.planning.react.middleware import ReActMiddleware, ReActModelMiddleware
from agent_core.planning.react.policy import ReActContext, ReActPolicy
from agent_core.planning.react.state import ReActState
from agent_core.planning.react.tool_access import ToolAccess


def compile_react_graph(
    model: Any,
    tools: Sequence[BaseTool],
    *,
    context_schema: type[Any] = ReActContext,
    stream_chat_message: Callable[..., Any],
    policy: ReActPolicy,
    middleware: Sequence[AgentMiddleware] = (),
    memory: Any = None,
    checkpointer: Any = None,
    store: Any = None,
    response_format: Any = None,
    publish_final: bool = True,
    system_prompt: str | None = None,
    context_summarizer: Any = None,
    native_authorizer: Any = None,
) -> Any:
    """Build one Run; LangChain owns model/tool routing and natural completion.

    Extra middleware uses LangChain's public API. The innermost model adapter
    preserves the Run's streaming cancellation and usage accounting. Serial
    execution preserves the governed executor's evidence ordering and limits.
    """
    if tools and not callable(getattr(model, "bind_tools", None)):
        raise ToolCallingUnsupportedError(policy.text("tool_calling_unsupported"))
    adapter = StreamingChatModelAdapter(delegate=model, collect=stream_chat_message)
    components = None
    if memory is not None:
        from agent_core.memory import build_memory_components
        components = build_memory_components(memory)
    memory_middleware = components.middleware if components else ()
    memory_tools = components.tools if components else ()
    access = ToolAccess(policy, metadata=components.tool_metadata if components else None)
    contributed = [*tools, *memory_tools]
    for component in (*memory_middleware, *middleware):
        contributed.extend(getattr(component, "tools", ()))
    access.register(contributed)
    model_calls = ReActModelMiddleware(policy, stream_chat_message, summarizer=context_summarizer, access=access)
    return create_agent(
        model=adapter,
        tools=[*tools, *memory_tools, access.search_tool],
        middleware=[ReActMiddleware(policy, model_calls, components=components, access=access, publish_final=publish_final, native_authorizer=native_authorizer),
                    *memory_middleware, *middleware, ToolHistoryMiddleware(), model_calls],
        state_schema=ReActState,
        context_schema=context_schema,
        checkpointer=checkpointer,
        store=store,
        response_format=response_format,
        system_prompt=system_prompt,
    )


__all__ = ["compile_react_graph"]
