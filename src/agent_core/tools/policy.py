"""Composable authorization policies applied before exposing or executing tools."""

from collections.abc import Sequence
from agent_core.runtime.streaming import call_port, run_sync

from agent_core.tools.contracts import AuthorizationDecision, ToolAuthorizer, ToolRuntime, ToolSpec


class AllowRegisteredTools:
    """The application explicitly supplied the executable catalog."""

    def authorize(self, definition: ToolSpec, runtime: ToolRuntime) -> AuthorizationDecision:
        return AuthorizationDecision(allowed=True)


class AllOfToolAuthorizer:
    """Require every supplied policy to allow the tool; preserve the first denial."""

    def __init__(self, policies: Sequence[ToolAuthorizer]) -> None:
        self._policies = tuple(policies)

    async def aauthorize(self, definition: ToolSpec, runtime: ToolRuntime) -> AuthorizationDecision:
        for policy in self._policies:
            decision = await call_port(policy, "authorize", definition, runtime)
            if not decision.allowed:
                return decision
        return AuthorizationDecision(allowed=True)

    def authorize(self, definition: ToolSpec, runtime: ToolRuntime) -> AuthorizationDecision:
        return run_sync(self.aauthorize(definition, runtime))
