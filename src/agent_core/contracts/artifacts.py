"""Optional evidence and structured products returned by a workflow."""
from dataclasses import dataclass, field
from typing import Any, Protocol


class Evidence(Protocol):
    reference_id: str


@dataclass(frozen=True)
class Artifact:
    kind: str
    value: Any
    metadata: dict[str, Any] = field(default_factory=dict)
