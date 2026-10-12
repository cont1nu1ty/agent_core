"""Explicit host resources used by the reusable memory components."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any
from agent_core.runtime.streaming import call_port, run_sync

from langgraph.store.base import BaseStore

from agent_core.memory.contracts import AsyncHistoryArchive, AsyncMemoryServices, HistoryArchive, MemoryRelease, MemoryScope, MemoryServices, MemorySnapshot


@dataclass(frozen=True, slots=True)
class MemoryLimits:
    max_notes_characters: int = 12_000
    max_file_characters: int = 64_000
    max_published_characters: int = 16_000
    max_history_characters: int = 16_000
    max_history_results: int = 10
    max_active_skills: int = 64
    confirmed_estimated_tokens: int = 2_048
    learned_estimated_tokens: int = 1_024

    def __post_init__(self) -> None:
        if any(getattr(self, name) < 1 for name in self.__dataclass_fields__):
            raise ValueError("memory limits must be positive")


@dataclass(slots=True)
class MemoryBinding:
    scope: MemoryScope
    store: BaseStore
    services: MemoryServices | AsyncMemoryServices
    history: HistoryArchive | AsyncHistoryArchive | None = None
    limits: MemoryLimits = field(default_factory=MemoryLimits)
    snapshot: MemorySnapshot | None = field(default=None, init=False)

    async def arefresh(self) -> MemorySnapshot:
        snapshot = await call_port(self.services, "snapshot", self.scope)
        if self.scope.project_id is None and snapshot.project is not None:
            raise ValueError("project memory requires a trusted project scope")
        self.snapshot = snapshot
        return snapshot

    def refresh(self) -> MemorySnapshot:
        return run_sync(self.arefresh())


def memory_binding_checkpoint(binding: MemoryBinding) -> dict[str, Any]:
    """Record the immutable publication view selected for a model decision."""
    if binding.snapshot is None:
        raise ValueError("memory publication has not been resolved")
    return {"scope": asdict(binding.scope), "snapshot": asdict(binding.snapshot)}


def restore_memory_binding(binding: MemoryBinding, state: Mapping[str, Any]) -> bool:
    """Restore a saved decision's namespaces before resuming its tools.

    This does not authorize stale reads: the tool policy still checks current
    release revisions, and the next before_model hook resolves a fresh view.
    """
    saved = state.get("memory_release_snapshot")
    if saved is None:
        return False
    if saved.get("scope") != asdict(binding.scope):
        raise PermissionError("saved memory scope differs from the resumed Run")

    def release(value: Mapping[str, Any]) -> MemoryRelease:
        return MemoryRelease(str(value["revision"]), tuple(value["memory_namespace"]), tuple(value["skills_namespace"]),
                             bool(value.get("layered", False)), tuple(value.get('memory_source_ids', ())),
                             bool(value.get('exposure_known', False)))

    payload = saved["snapshot"]
    binding.snapshot = MemorySnapshot(release(payload["user"]),
        release(payload["project"]) if payload.get("project") is not None else None,
        tuple(payload.get("revoked_source_ids", ())))
    return True
