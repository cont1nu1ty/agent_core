"""Task-note conventions over Deep Agents' checkpointed files channel."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import PurePosixPath
import re
from typing import Any

from deepagents.backends.utils import file_data_to_string


TASK_NOTES_PATH = "/notes/TASK.md"
TASK_NOTE_SECTIONS = ("目标", "约束", "关键结论", "已排除方案", "证据位置", "下一步", "任务完成判断及依据")
TASK_NOTES_TEMPLATE = "# 任务笔记\n\n" + "\n\n".join(f"## {title}\n待记录。" for title in TASK_NOTE_SECTIONS) + "\n"
_PLACEHOLDER = re.compile(r"\s*(?:[-*]\s*)?(?:待记录|待补充|尚未记录|未记录|暂无|无)?[。.!！]?\s*")
_SOURCE_REFERENCE = re.compile(
    r"(?<![\w:])(?:memory:[A-Za-z0-9_.-]+:v[1-9]\d*|"
    r"(?!memory:)[A-Za-z0-9_.-]+:[A-Za-z0-9_.-]+)(?![\w:])"
)


def _has_reference_location(content: str) -> bool:
    if _SOURCE_REFERENCE.search(content):
        return True
    for match in re.finditer(r"/(?:notes|scratch|drafts|memory|skills)/[^\s`<>\"']+", content):
        path = match.group().rstrip(".,;:，。；：)")
        if "\\" in path or ".." in PurePosixPath(path).parts:
            continue
        if path == TASK_NOTES_PATH or path.startswith(("/scratch/", "/drafts/")):
            return True
        if path in {"/memory/user/AGENTS.md", "/memory/project/AGENTS.md"}:
            return True
        if path.startswith(("/skills/user/", "/skills/project/")) and path.endswith("/SKILL.md"):
            return True
    return False


def validate_task_notes(content: str, *, max_characters: int = 12_000) -> None:
    """Check the small note contract without implementing a Markdown parser."""
    if len(content) > max_characters:
        raise ValueError("任务笔记超过长度上限，请保留结论和证据位置")
    missing = [name for name in TASK_NOTE_SECTIONS if not re.search(rf"^## {re.escape(name)}\s*$", content, re.MULTILINE)]
    if missing:
        raise ValueError("任务笔记缺少必要小节：" + "、".join(missing))
    # Only our fixed template is split here, not arbitrary Markdown. This checks
    # reference location syntax; History/Memory reads still authorize sources,
    # and a syntactically valid citation does not prove a conclusion is correct.
    sections = re.split(r"^## (" + "|".join(map(re.escape, TASK_NOTE_SECTIONS)) + r")\s*$", content, flags=re.MULTILINE)
    contents = dict(zip(sections[1::2], sections[2::2]))
    if not _PLACEHOLDER.fullmatch(contents["关键结论"]) and not _has_reference_location(contents["证据位置"]):
        raise ValueError("任务笔记的关键结论需要证据位置：请填写 source_id、memory:id:vN 或可回读虚拟路径")


def task_notes_text(state: Mapping[str, Any]) -> str:
    file_data = state.get("files", {}).get(TASK_NOTES_PATH)
    if not file_data:
        return ""
    if file_data.get("encoding", "utf-8") != "utf-8":
        raise ValueError("task notes must be UTF-8 text")
    return file_data_to_string(file_data)


def task_memory_snapshot(state: Mapping[str, Any]) -> dict[str, Any]:
    """Project only user-facing task memory, never the full graph state."""
    todos = [
        {"content": item["content"], "status": item["status"]}
        for item in state.get("todos", [])
        if isinstance(item, Mapping) and isinstance(item.get("content"), str)
        and item.get("status") in {"pending", "in_progress", "completed"}
    ]
    snapshot = {"notesMarkdown": task_notes_text(state), "todos": todos,
                "sourceIds": list(state.get("task_source_ids", []))}
    if state.get("plan_state"):
        from agent_core.planning.plan_execute import project_plan
        snapshot["plan"] = project_plan(state["plan_state"])
    return snapshot


def inherit_task_memory(state: Mapping[str, Any]) -> dict[str, Any]:
    """Carry only the task note and plan into a new Run of the same task."""
    update: dict[str, Any] = {
        "todos": deepcopy(task_memory_snapshot(state)["todos"]),
        "task_source_ids": list(state.get("task_source_ids", [])),
    }
    file_data = state.get("files", {}).get(TASK_NOTES_PATH)
    if file_data is not None:
        # Preserve the official FileData value rather than maintain another codec.
        update["files"] = {TASK_NOTES_PATH: deepcopy(file_data)}
    if state.get("plan_state"):
        # Plan-level accounting and facts survive user turns. Run-local tool
        # IDs, messages and private execution state remain owned by the Run.
        update["plan_state"] = deepcopy(state["plan_state"])
        # A plan can finish steps in several Runs. Keep its public evidence
        # objects and numbering so old result citations cannot point at new I/O.
        update["references"] = deepcopy(state.get("references", []))
    return update
