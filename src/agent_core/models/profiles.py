"""Explicit settings for a model without environment or host dependencies."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelProfile:
    """OpenAI-compatible model settings; callers own configuration loading."""

    model: str
    base_url: str
    api_key: str = field(repr=False)
    provider: str = "openai"
    temperature: float = 0.2
    timeout_seconds: float = 120.0
    max_tokens: int | None = None
