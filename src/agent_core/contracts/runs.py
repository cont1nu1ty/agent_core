"""Inputs and results of one invocation, without a product session or database."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ResumeDecision:
    """Host-authorized decisions bound to the exact saved interrupt IDs."""

    checkpoint_id: str
    decisions: dict[str, Any]

    def __post_init__(self) -> None:
        import json
        if not self.checkpoint_id or not self.decisions or any(not key for key in self.decisions):
            raise ValueError("Resume decisions require a checkpoint and interrupt identities")
        object.__setattr__(self, "decisions", json.loads(json.dumps(self.decisions, allow_nan=False)))


@dataclass(frozen=True)
class RunRequest:
    run_id: str
    workflow_id: str
    input: str
    attempt: int = 1
    resume: bool = False
    checkpoint_id: str | None = None
    binding_fingerprint: str | None = None
    resume_decision: ResumeDecision | None = None

    def __post_init__(self) -> None:
        if self.resume_decision is not None and not self.resume:
            raise ValueError("Resume decisions require explicit resume")


@dataclass(frozen=True)
class RunResult:
    output: Any = None
    # Explicit Artifact products and host-owned Evidence references coexist.
    # Only Artifact values produce artifact.ready; references keep their
    # existing host-specific validation and reference event projection.
    artifacts: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
