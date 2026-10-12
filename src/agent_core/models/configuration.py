"""Pure validation of explicitly supplied model settings; never reads environment."""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from agent_core.contracts.errors import ModelConfigurationError
from agent_core.models.profiles import ModelProfile


def model_profile_from_mapping(values: Mapping[str, Any]) -> ModelProfile:
    """Normalize a host's explicit values into the supported transport profile."""
    def clean(name: str) -> str:
        value = values.get(name)
        return str(value).strip() if value is not None else ""

    missing = [name for name in ("model", "base_url", "api_key") if not clean(name)]
    if missing:
        raise ModelConfigurationError("model configuration is incomplete; missing " + ", ".join(missing))
    provider = clean("provider").lower() or "openai"
    if provider in {"generic", "openai-compatible", "openai_compatible"}:
        provider = "openai"
    if provider != "openai":
        raise ModelConfigurationError("model provider must use the supported openai-compatible transport")
    base_url = clean("base_url")
    try:
        parsed = urlparse(base_url)
        # Accessing port validates both numeric syntax and its legal range.
        _ = parsed.port
    except ValueError as error:
        raise ModelConfigurationError("model base_url must be a valid HTTP(S) URL") from error
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ModelConfigurationError("model base_url must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ModelConfigurationError("model base_url must not contain credentials, query or fragment")

    def number(name: str, default: float, minimum: float, maximum: float | None = None) -> float:
        try:
            value = float(clean(name) or default)
        except (TypeError, ValueError) as error:
            raise ModelConfigurationError(f"model {name} must be a number") from error
        if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
            raise ModelConfigurationError(f"model {name} is outside its supported range")
        return value

    max_tokens = None
    if raw := clean("max_tokens"):
        try:
            max_tokens = int(raw)
        except ValueError as error:
            raise ModelConfigurationError("model max_tokens must be an integer") from error
        if max_tokens < 1:
            raise ModelConfigurationError("model max_tokens must be >= 1")
    return ModelProfile(
        model=clean("model"), provider=provider, base_url=base_url.rstrip("/"), api_key=clean("api_key"),
        temperature=number("temperature", 0.2, 0.0, 2.0),
        timeout_seconds=number("timeout_seconds", 120.0, 1.0), max_tokens=max_tokens,
    )


def model_connection_snapshot(profile: ModelProfile) -> dict[str, Any]:
    """Return non-secret execution settings. Credential rotation is independent."""
    return {name: getattr(profile, name) for name in (
        "model", "provider", "base_url", "temperature", "timeout_seconds", "max_tokens",
    )}
