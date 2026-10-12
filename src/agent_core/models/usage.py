"""Normalize usage from one completed, merged provider message."""
from collections.abc import Mapping
from typing import Any


def _token_count(value: Any) -> int | None:
    # Do not turn booleans, invalid values, or absent provider usage into zero.
    return value if type(value) is int and value >= 0 else None


def normalize_usage(message: Any) -> dict[str, Any]:
    metadata = getattr(message, "usage_metadata", None)
    if isinstance(metadata, Mapping):
        counts = {
            name: _token_count(metadata.get(name))
            for name in ("input_tokens", "output_tokens", "total_tokens")
        }
        if any(value is not None for value in counts.values()):
            return {**counts, "source": "reported"}
    response_metadata = getattr(message, "response_metadata", None)
    raw_usage = (
        response_metadata.get("token_usage")
        if isinstance(response_metadata, Mapping)
        else None
    )
    if isinstance(raw_usage, Mapping):
        counts = {
            "input_tokens": _token_count(raw_usage.get("prompt_tokens")),
            "output_tokens": _token_count(raw_usage.get("completion_tokens")),
            "total_tokens": _token_count(raw_usage.get("total_tokens")),
        }
        if any(value is not None for value in counts.values()):
            return {**counts, "source": "reported"}
    return {}
