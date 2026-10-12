"""Scoped memory ports; hosts own persistence, identity and publication policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


MemoryTarget = Literal["user", "project"]
MemoryKind = Literal["memory", "skill", "workflow_proposal"]
MemoryStatus = Literal["candidate", "confirmed", "learned", "revoked"]
MemoryUsageKind = Literal["injected", "retrieved", "model_reported_used", "corrected"]


class MemoryOperationError(ValueError):
    """A recoverable memory operation refusal with a safe public message."""

    def __init__(self, message: str = "记忆操作未完成，请核对记录和来源", *,
                 code: str = "memory_operation_rejected", retryable: bool = False) -> None:
        super().__init__(message)
        self.public_message = message
        self.code = code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class MemoryScope:
    tenant_id: str
    user_id: str
    project_id: str | None
    session_id: str
    task_id: str
    run_id: str

    def __post_init__(self) -> None:
        for name in ("tenant_id", "user_id", "session_id", "task_id", "run_id"):
            if not getattr(self, name).strip():
                raise ValueError(f"memory scope requires {name}")
        if self.project_id is not None and not self.project_id.strip():
            raise ValueError("project_id cannot be blank")


@dataclass(frozen=True, slots=True)
class MemoryRelease:
    revision: str
    memory_namespace: tuple[str, ...]
    skills_namespace: tuple[str, ...]
    layered: bool = False
    memory_source_ids: tuple[str, ...] = ()
    exposure_known: bool = False


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    user: MemoryRelease
    project: MemoryRelease | None = None
    revoked_source_ids: tuple[str, ...] = ()

    @property
    def revisions(self) -> dict[str, str]:
        result = {"user": self.user.revision}
        if self.project is not None:
            result["project"] = self.project.revision
        return result


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    record_id: str
    kind: MemoryKind
    target: MemoryTarget
    title: str
    content: str
    version: int
    status: MemoryStatus
    source_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryMutationRequest:
    action: Literal["remember", "correct", "forget", "propose_skill", "propose_workflow", "confirm"]
    target: MemoryTarget
    content: str = ""
    title: str = ""
    record_id: str | None = None
    expected_version: int | None = None
    source_ids: tuple[str, ...] = ()
    provenance: Literal["user", "inferred"] = "inferred"


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    record: MemoryRecord
    changed: bool = True
    snapshot: MemorySnapshot | None = None


@dataclass(frozen=True, slots=True)
class LearnedMemoryProposal:
    """Evidence-backed experience proposed by a host background workflow.

    It is never a user confirmation and is not accepted by model mutation tools.
    The host owns evidence quality; publication rechecks scope and source liveness.
    """

    content: str
    title: str = ""
    source_ids: tuple[str, ...] = ()
    applicability: str = ""
    evidence_task_ids: tuple[str, ...] = ()
    category: Literal["procedure", "preference", "project_fact"] = "procedure"


class MemoryServices(Protocol):
    """Every operation must authorize scope and source IDs on the host.

    A model label is not confirmation. Hosts verify user-authored evidence before
    publishing ``provenance=user`` records; model-originated mutations remain
    candidates. Only a separate host background publisher can activate learned
    experience, which never becomes user confirmation automatically.
    Missing source IDs may be resolved by the host from the current user Turn.
    ``operation_id`` is a durable tool-call identity for idempotent replay.
    """

    def snapshot(self, scope: MemoryScope) -> MemorySnapshot: ...

    def list_records(
        self, scope: MemoryScope, *, target: MemoryTarget,
        kind: MemoryKind | None = None, status: MemoryStatus | None = None,
        query: str = "", limit: int = 20, offset: int = 0,
    ) -> Sequence[MemoryRecord]: ...

    def read_record(
        self, scope: MemoryScope, record_id: str, *, version: int | None = None,
    ) -> MemoryRecord | None: ...

    def mutate(
        self, scope: MemoryScope, change: MemoryMutationRequest, *, operation_id: str,
    ) -> MemoryMutationResult: ...

    def valid_sources(
        self, scope: MemoryScope, source_ids: Sequence[str], *, target: MemoryTarget = "project",
    ) -> Sequence[str]:
        """Return sources still authorized for derivation, including tombstone checks."""
        ...

    def invalidated_sources(self, scope: MemoryScope) -> Sequence[str]:
        """Return complete durable invalidations; snapshot warnings may be bounded."""
        ...

    def record_usage(
        self, scope: MemoryScope, record_id: str, version: int, *,
        kind: MemoryUsageKind, operation_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class HistoryEntry:
    source_id: str
    run_id: str
    sequence: int
    role: Literal["user", "assistant", "tool"]
    content: str
    call_id: str | None = None
    tool_name: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class HistoryArchive(Protocol):
    """Canonical safe history, distinct from the recent-QA input projection."""

    def append(self, scope: MemoryScope, entries: Sequence[HistoryEntry]) -> None: ...

    def search(
        self, scope: MemoryScope, query: str, *, limit: int = 10,
        task_id: str | None = None,
    ) -> Sequence[HistoryEntry]:
        """Find authorized entries containing a literal substring in content.

        Query terms are not split or interpreted as predicates. Tool-name
        metadata is returned with entries but is not a searchable text index.
        No matches return an empty sequence, not an authorization failure.
        """
        ...

    def read(
        self, scope: MemoryScope, source_ids: Sequence[str], *, max_characters: int = 16_000,
        offset_characters: int = 0,
    ) -> Sequence[HistoryEntry]:
        """Read bounded text; a nonzero offset requires a single source.

        Entries retain their original identity. Metadata includes
        ``offset_characters``, ``original_characters`` and ``truncated`` so a
        model can continue reading without treating excerpts as full evidence.
        """
        ...


class AsyncMemoryServices(Protocol):
    """Every operation must authorize scope and source IDs on the host.

    A model label is not confirmation. Hosts verify user-authored evidence before
    publishing ``provenance=user`` records; model-originated mutations remain
    candidates. Only a separate host background publisher can activate learned
    experience, which never becomes user confirmation automatically.
    Missing source IDs may be resolved by the host from the current user Turn.
    ``operation_id`` is a durable tool-call identity for idempotent replay.
    """

    async def asnapshot(self, scope: MemoryScope) -> MemorySnapshot: ...

    async def alist_records(
        self, scope: MemoryScope, *, target: MemoryTarget,
        kind: MemoryKind | None = None, status: MemoryStatus | None = None,
        query: str = "", limit: int = 20, offset: int = 0,
    ) -> Sequence[MemoryRecord]: ...

    async def aread_record(
        self, scope: MemoryScope, record_id: str, *, version: int | None = None,
    ) -> MemoryRecord | None: ...

    async def amutate(
        self, scope: MemoryScope, change: MemoryMutationRequest, *, operation_id: str,
    ) -> MemoryMutationResult: ...

    async def avalid_sources(
        self, scope: MemoryScope, source_ids: Sequence[str], *, target: MemoryTarget = "project",
    ) -> Sequence[str]:
        """Return sources still authorized for derivation, including tombstone checks."""
        ...

    async def ainvalidated_sources(self, scope: MemoryScope) -> Sequence[str]:
        """Return complete durable invalidations; snapshot warnings may be bounded."""
        ...

    async def arecord_usage(
        self, scope: MemoryScope, record_id: str, version: int, *,
        kind: MemoryUsageKind, operation_id: str,
    ) -> None: ...


class AsyncHistoryArchive(Protocol):
    """Canonical safe history, distinct from the recent-QA input projection."""

    async def aappend(self, scope: MemoryScope, entries: Sequence[HistoryEntry]) -> None: ...

    async def asearch(
        self, scope: MemoryScope, query: str, *, limit: int = 10,
        task_id: str | None = None,
    ) -> Sequence[HistoryEntry]:
        """Find authorized entries containing a literal substring in content.

        Query terms are not split or interpreted as predicates. Tool-name
        metadata is returned with entries but is not a searchable text index.
        No matches return an empty sequence, not an authorization failure.
        """
        ...

    async def aread(
        self, scope: MemoryScope, source_ids: Sequence[str], *, max_characters: int = 16_000,
        offset_characters: int = 0,
    ) -> Sequence[HistoryEntry]:
        """Read bounded text; a nonzero offset requires a single source.

        Entries retain their original identity. Metadata includes
        ``offset_characters``, ``original_characters`` and ``truncated`` so a
        model can continue reading without treating excerpts as full evidence.
        """
        ...
