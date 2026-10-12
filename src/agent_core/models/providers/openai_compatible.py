"""Request parameter mapping for OpenAI-compatible gateways."""
from typing import Any
from urllib.parse import urlparse

from agent_core.models.profiles import ModelProfile

_BIGMODEL_OPENAI_GATEWAY_HOSTS = {"open.bigmodel.cn"}
_BIGMODEL_FORCED_THINKING_MODELS = {"glm-5.3", "glm-5.3-flash"}


def supports_required_tool_choice(profile: ModelProfile) -> bool:
    """BigModel documents automatic tool selection as its only supported mode."""
    return urlparse(profile.base_url or "").hostname not in _BIGMODEL_OPENAI_GATEWAY_HOSTS


def chat_options(profile: ModelProfile) -> dict[str, Any]:
    """Preserve the existing generic and BigModel request contracts."""
    options: dict[str, Any] = {
        "model": profile.model,
        "api_key": profile.api_key,
        "base_url": profile.base_url,
        "temperature": profile.temperature,
        "timeout": profile.timeout_seconds,
        # The explicitly owned httpx clients below preserve proxy environment
        # handling; LangChain must not replace their transport for socket tuning.
        "http_socket_options": (),
    }
    endpoint_host = urlparse(profile.base_url or "").hostname
    if endpoint_host in _BIGMODEL_OPENAI_GATEWAY_HOSTS:
        # Earlier GLM models support disabling reasoning. GLM-5.3 models
        # require it; select their supported low effort for bounded agent calls
        # rather than sending an unsupported disabled flag or defaulting to max.
        extra_body: dict[str, Any] = {
            "thinking": {"type": "disabled"},
            # GLM requires this companion flag when ``stream=True`` and the
            # response may contain incremental tool calls.
            "tool_stream": True,
        }
        if profile.model.casefold() in _BIGMODEL_FORCED_THINKING_MODELS:
            # Clear historical reasoning using the API's standard default;
            # this does not establish reasoning replay support within a turn.
            extra_body["thinking"] = {"type": "enabled", "clear_thinking": True}
            options["reasoning_effort"] = "low"
        if profile.max_tokens is not None:
            # BigModel's OpenAI-compatible Chat Completions contract names
            # this parameter ``max_tokens``. ChatOpenAI otherwise normalizes
            # it to OpenAI's newer ``max_completion_tokens`` field.
            extra_body["max_tokens"] = profile.max_tokens
        options["extra_body"] = extra_body
    elif profile.max_tokens is not None:
        options["max_completion_tokens"] = profile.max_tokens
    return options
