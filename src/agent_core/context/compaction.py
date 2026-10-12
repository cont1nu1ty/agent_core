"""Select complete recent history pairs under a deterministic budget."""
from collections.abc import Sequence

from agent_core.memory.history import HistoryMessage


def compact_history(
    history: Sequence[HistoryMessage], *, max_pairs: int = 6, max_characters: int = 16_000,
) -> tuple[HistoryMessage, ...]:
    """Stop at the first over-budget pair; return chronological history."""
    selected: list[HistoryMessage] = []
    total = 0
    for pair in reversed(history):
        pair_length = len(pair.user_message) + len(pair.assistant_answer)
        if total + pair_length > max_characters or len(selected) >= max_pairs:
            break
        selected.append(pair)
        total += pair_length
    return tuple(reversed(selected))
