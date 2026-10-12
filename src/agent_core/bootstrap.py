"""Explicit assembly for embedded applications and standalone callers."""
from agent_core.runtime.service import AgentService


def build_agent(*, workflows, model_factory=None, settings=None, usage_observer=None, checkpointer=None, store=None,
                model_ownership="owned", persistence_factory=None):
    return AgentService(
        workflows=workflows, model_factory=model_factory,
        settings=settings, usage_observer=usage_observer, checkpointer=checkpointer, store=store,
        model_ownership=model_ownership, persistence_factory=persistence_factory,
    )
