"""Usage observer for integrations without an exporter."""
from agent_core.observability.contracts import UsageRecord


class NoopUsageObserver:
    def record(self, usage: UsageRecord) -> None:
        pass
