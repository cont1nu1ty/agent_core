"""Select an authorized catalog subset without mixing selection into execution."""

from __future__ import annotations

from collections.abc import Collection
from typing import TYPE_CHECKING

from agent_core.tools.contracts import ToolAuthorizer, ToolRuntime, ToolSpec
from agent_core.tools.policy import AllowRegisteredTools

if TYPE_CHECKING:
    from agent_core.tools.registry import ToolRegistry


class CatalogToolSelector:
    """Deterministic coarse selection; model-based relevance can be added separately."""

    def select(
        self,
        registry: ToolRegistry,
        runtime: ToolRuntime,
        *,
        authorizer: ToolAuthorizer | None = None,
        category: str | None = None,
        version: str | None = None,
        tags: Collection[str] = (),
        names: Collection[str] | None = None,
        limit: int | None = None,
    ) -> tuple[ToolSpec, ...]:
        if limit is not None and limit < 0:
            raise ValueError("tool selection limit cannot be negative")
        if limit == 0:
            return ()
        policy = authorizer or AllowRegisteredTools()
        allowed_names = frozenset(names) if names is not None else None
        selected: list[ToolSpec] = []
        for definition in registry.query(category=category, version=version, tags=tags):
            if allowed_names is not None and definition.name not in allowed_names:
                continue
            if not policy.authorize(definition, runtime).allowed:
                continue
            selected.append(definition)
            if limit is not None and len(selected) >= limit:
                break
        return tuple(selected)
