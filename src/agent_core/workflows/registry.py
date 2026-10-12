"""An explicit registry; no automatic imports, defaults or discovery."""
from collections.abc import Iterable
from agent_core.workflows.contracts import WorkflowDefinition


class WorkflowRegistry:
    def __init__(self, definitions: Iterable[WorkflowDefinition] = ()) -> None:
        self._definitions: dict[str, WorkflowDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: WorkflowDefinition) -> None:
        if definition.name in self._definitions:
            raise ValueError(f"duplicate workflow: {definition.name}")
        self._definitions[definition.name] = definition

    def resolve(self, name: str) -> WorkflowDefinition:
        try:
            return self._definitions[name]
        except KeyError as error:
            raise ValueError(f"unknown workflow: {name}") from error

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)
