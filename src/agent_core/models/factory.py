"""Construct task-owned model clients and clean up failed construction."""
from __future__ import annotations

from threading import Thread

from agent_core.contracts.errors import ModelConfigurationError
from agent_core.models.base import ChatModel
from agent_core.models.configuration import model_connection_snapshot, model_profile_from_mapping
from agent_core.models.profiles import ModelProfile
from agent_core.models.providers.openai_compatible import (
    chat_options,
    supports_required_tool_choice,
)


def build_chat_model(profile: ModelProfile, *, options: dict | None = None) -> ChatModel:
    """Build a model from explicit settings without reading host configuration.

    The OpenAI-compatible provider owns the HTTP clients. Gateway-specific
    request parameters remain unchanged from the existing integration.
    """
    missing = [name for name in ("model", "base_url", "api_key") if not getattr(profile, name)]
    if missing:
        raise ModelConfigurationError("model configuration is incomplete; missing " + ", ".join(missing))

    profile = model_profile_from_mapping({**model_connection_snapshot(profile), "api_key": profile.api_key})
    supplied_options = options or {}
    if set(supplied_options) - {"streaming", "stream_usage", "use_responses_api", "reasoning_effort", "temperature"}:
        raise ModelConfigurationError("model overrides cannot change its connection or credentials")
    if "temperature" in supplied_options and supplied_options["temperature"] is not None:
        validated = model_profile_from_mapping({
            **model_connection_snapshot(profile), "api_key": profile.api_key,
            "temperature": supplied_options["temperature"],
        })
        supplied_options = {**supplied_options, "temperature": validated.temperature}
    options = {**chat_options(profile), **supplied_options}

    # Keep dependencies lazy: importing profiles must not initialize clients.
    # Explicit per-run clients also avoid LangChain's process-wide cached AsyncClient,
    # which cannot safely move between the agent's isolated event loops.
    import httpx
    from langchain_openai import ChatOpenAI

    sync_http_client = httpx.Client(timeout=profile.timeout_seconds)
    try:
        async_http_client = httpx.AsyncClient(timeout=profile.timeout_seconds)
    except Exception:
        sync_http_client.close()
        raise
    options["http_client"] = sync_http_client
    options["http_async_client"] = async_http_client
    try:
        chat_model = ChatOpenAI(**options)
    except Exception:
        sync_http_client.close()

        async def close_async_client() -> None:
            await async_http_client.aclose()

        import asyncio

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                asyncio.run(close_async_client())
            except Exception:
                pass
        else:
            # This builder is synchronous, but keeping its failure cleanup safe
            # inside an async caller prevents ``asyncio.run`` from masking the
            # original model-construction exception.
            def close_in_thread() -> None:
                try:
                    asyncio.run(close_async_client())
                except Exception:
                    pass

            cleanup_thread = Thread(target=close_in_thread, daemon=True)
            cleanup_thread.start()
            cleanup_thread.join()
        raise
    return ChatModel(
        chat_model=chat_model,
        _sync_http_client=sync_http_client,
        _async_http_client=async_http_client,
        max_tokens=profile.max_tokens,
        supports_required_tool_choice=supports_required_tool_choice(profile),
    )
