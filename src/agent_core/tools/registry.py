"""Explicit tool catalog; applications own catalog composition and policy."""

from collections.abc import Collection, Iterable, Iterator

from agent_core.contracts.errors import ToolNotRegisteredError
from agent_core.tools.contracts import ToolAuthorizer, ToolRuntime, ToolSpec
from agent_core.tools.selection import CatalogToolSelector


class ToolRegistry:
    def __init__(self, definitions: Iterable[ToolSpec]) -> None:
        self._definitions = tuple(definitions)
        self._by_name: dict[str, ToolSpec] = {}
        self._by_category: dict[str, set[str]] = {}
        self._by_version: dict[str, set[str]] = {}
        self._by_tag: dict[str, set[str]] = {}
        for definition in self._definitions:
            if definition.name in self._by_name:
                raise ValueError(f"duplicate tool definition: {definition.name}")
            self._by_name[definition.name] = definition
            self._by_category.setdefault(definition.category, set()).add(definition.name)
            self._by_version.setdefault(definition.version, set()).add(definition.name)
            for tag in definition.tags:
                self._by_tag.setdefault(tag, set()).add(definition.name)

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self._definitions)

    def __len__(self) -> int:
        return len(self._definitions)

    @property
    def definitions(self) -> tuple[ToolSpec, ...]:
        return self._definitions

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(definition.name for definition in self._definitions)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._by_name[name]
        except KeyError as error:
            raise ToolNotRegisteredError() from error

    def available(self, runtime: ToolRuntime, *, authorizer: ToolAuthorizer | None = None) -> tuple[ToolSpec, ...]:
        return CatalogToolSelector().select(self, runtime, authorizer=authorizer)

    def query(self, *, category: str | None = None, version: str | None = None, tags: Collection[str] = ()) -> tuple[ToolSpec, ...]:
        """Intersect catalog indexes, returning matches in stable registration order."""
        candidates = set(self._by_name)
        if category is not None:
            candidates.intersection_update(self._by_category.get(category, ()))
        if version is not None:
            candidates.intersection_update(self._by_version.get(version, ()))
        for tag in tags:
            candidates.intersection_update(self._by_tag.get(tag, ()))
        return tuple(item for item in self._definitions if item.name in candidates)
