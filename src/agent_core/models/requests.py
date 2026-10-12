"""Per-call output limits that preserve explicitly configured gateway options."""
from collections.abc import Mapping
from dataclasses import replace
from typing import Any


def output_limit_kwargs(model: Any, limit: int) -> dict[str, Any]:
    """Override both standard and explicit extra-body output limits.

    Some compatible gateways require max_tokens in extra_body. That body is
    merged after standard parameters by the provider SDK, so a bare per-call
    max_tokens would leave the host profile's larger limit in effect.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError("model output limit must be a positive integer")
    result: dict[str, Any] = {"max_tokens": limit}
    provider = getattr(model, "chat_model", model)
    extra = getattr(provider, "extra_body", None)
    if isinstance(extra, Mapping) and any(key in extra for key in ("max_tokens", "max_completion_tokens")):
        result["extra_body"] = {**extra, **{key: limit for key in
            ("max_tokens", "max_completion_tokens") if key in extra}}
    return result


def bounded_call_model(model: Any) -> Any:
    """Disable hidden SDK retries for explicitly counted auxiliary calls.

    Public Pydantic model_copy and OpenAI with_options create a request view
    sharing the original owned HTTP clients. The caller must still close the
    original resource; ordinary foreground calls retain their configured retry
    behavior. Other injected model implementations own their retry contract.
    """
    from langchain_openai import ChatOpenAI
    from agent_core.models.base import ChatModel

    provider = model.chat_model if isinstance(model, ChatModel) else model
    if not isinstance(provider, ChatOpenAI):
        return model
    updates: dict[str, Any] = {"max_retries": 0}
    for root_name, endpoint_name in (("root_client", "client"), ("root_async_client", "async_client")):
        original = getattr(provider, root_name)
        if original is not None:
            client = original.with_options(max_retries=0)
            updates[root_name] = client
            updates[endpoint_name] = client.chat.completions
    view = provider.model_copy(update=updates)
    return replace(model, chat_model=view) if isinstance(model, ChatModel) else view
