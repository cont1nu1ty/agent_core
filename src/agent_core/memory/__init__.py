"""Conversation history ports and immutable snapshots."""
from agent_core.memory.history import AsyncHistoryReader, HistoryMessage, HistoryReader
from agent_core.memory.stores.snapshot import SnapshotHistory
from agent_core.memory.binding import MemoryBinding, MemoryLimits, restore_memory_binding
from agent_core.memory.components import MemoryComponents, build_memory_components
from agent_core.memory.checkpoints import aread_task_checkpoint, read_task_checkpoint
from agent_core.memory.contracts import (
    AsyncHistoryArchive, AsyncMemoryServices, HistoryArchive, HistoryEntry, LearnedMemoryProposal, MemoryKind, MemoryMutationRequest, MemoryMutationResult, MemoryOperationError,
    MemoryRecord, MemoryRelease, MemoryScope, MemoryServices, MemorySnapshot,
    MemoryStatus, MemoryTarget, MemoryUsageKind,
)
from agent_core.memory.notes import (
    TASK_NOTES_PATH, TASK_NOTES_TEMPLATE, inherit_task_memory, task_memory_snapshot,
    validate_task_notes,
)

__all__ = [
    "AsyncHistoryReader", "AsyncHistoryArchive", "AsyncMemoryServices", "aread_task_checkpoint",
    "HistoryMessage", "HistoryReader", "SnapshotHistory", "MemoryBinding", "MemoryLimits",
    "MemoryComponents", "build_memory_components", "HistoryArchive", "HistoryEntry",
    "MemoryKind", "MemoryMutationRequest", "MemoryMutationResult", "MemoryRecord",
    "MemoryRelease", "MemoryScope", "MemoryServices", "MemorySnapshot", "MemoryStatus",
    "MemoryTarget", "TASK_NOTES_PATH", "TASK_NOTES_TEMPLATE", "inherit_task_memory",
    "task_memory_snapshot", "validate_task_notes",
    "restore_memory_binding",
    "MemoryOperationError",
    "read_task_checkpoint",
    "LearnedMemoryProposal", "MemoryUsageKind",
]
