"""Trusted, schema-owned corrections for business input constraints.

A Pydantic model may declare ``tool_validation_hints`` as a ClassVar mapping
stable PydanticCustomError codes to these hints. Fields name top-level public
input properties. Messages must be constant guidance, never interpolated input
or exception text. Unlisted custom errors receive only generic safe feedback.
"""

from dataclasses import dataclass
from typing import get_args

from pydantic_core import ErrorType


STANDARD_VALIDATION_CODES = frozenset(get_args(ErrorType))


@dataclass(frozen=True, slots=True)
class ToolValidationHint:
    fields: tuple[str, ...]
    hint: str

    def __post_init__(self) -> None:
        if not isinstance(self.fields, tuple) or not 1 <= len(self.fields) <= 8 or any(
            not isinstance(name, str) or not 1 <= len(name) <= 160 for name in self.fields
        ):
            raise ValueError("validation hints require one to eight bounded public field names")
        if len(set(self.fields)) != len(self.fields):
            raise ValueError("validation hint fields must be unique")
        if not isinstance(self.hint, str) or not self.hint.strip() or len(self.hint) > 320:
            raise ValueError("validation hints require bounded, constant correction guidance")


__all__ = ["ToolValidationHint"]
