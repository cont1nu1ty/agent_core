"""Compose discovery with the existing model and tool governance boundaries."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any
from agent_core.runtime.streaming import call_port, call_async, run_sync, sync_facade

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agent_core.contracts.errors import AgentContextError
from agent_core.tools.contracts import ToolDiscoverySpec
from agent_core.tools.discovery import DiscoveryCoordinator, ExposurePlan, ToolCatalog, ToolDiscoveryError


class ToolAccess:
    """Run-local mirror; only checkpointed discovery/visibility is authoritative."""

    def __init__(self, policy: Any, *, metadata: Any = None) -> None:
        self.policy = policy
        self.metadata = dict(metadata or {})
        self.tools: tuple[Any, ...] = ()
        self.coordinator: DiscoveryCoordinator | None = None
        self._plan: ExposurePlan | None = None

        @tool
        async def search_tools(query: str, runtime: ToolRuntime[Any], category: str | None = None,
                         limit: int = 4) -> Command:
            """Search authorized tool capabilities, not business data. Loaded tools become callable next model turn."""
            context = runtime.context
            self.restore(runtime.state, context)
            if not context.settings.tool_discovery_enabled or self.coordinator is None:
                raise AgentContextError("Tool discovery is disabled")
            await self._refresh_authorization(context)
            plan = self._saved_plan(runtime.state)
            try:
                receipt = self.coordinator.search(query, category=category, limit=limit,
                    call_id=runtime.tool_call_id, plan=plan,
                    authorized=lambda entry: self._authorized(entry.tool, context),
                    available=lambda entry: self._available(entry.tool, context))
                failed = False
            except ToolDiscoveryError as error:
                receipt = {"ok": False, "errorCode": error.code, "message": error.public_message}
                failed = True
            context.cancellation.raise_if_cancelled()
            return Command(update={
                "messages": [ToolMessage(json.dumps(receipt, ensure_ascii=False, separators=(",", ":")),
                    tool_call_id=runtime.tool_call_id, name="search_tools", status="error" if failed else "success")],
                "tool_discovery": self.coordinator.snapshot(),
            })

        search_tools.func = sync_facade(search_tools.coroutine)
        self.search_tool = search_tools
        self._authorization = {}
        self.metadata["search_tools"] = {
            "version": "1.0.0", "category": "discovery", "source": "agent-core",
            "budget_group": "discovery", "effect": "control",
            "discovery": ToolDiscoverySpec(exposure="direct", required=True,
                aliases=("搜索工具", "发现能力"), when_to_use="需要本轮尚未加载的工具能力时搜索目录。"),
        }

    def register(self, tools: Any) -> None:
        names = [item.name for item in tools]
        if len(names) != len(set(names)) or "search_tools" in names:
            raise AgentContextError("Tool names must be unique; search_tools is reserved")
        for item in tools:
            contribution = (getattr(item, "extras", None) or {}).get("agent_core_catalog")
            if isinstance(contribution, Mapping):
                self.metadata.setdefault(item.name, dict(contribution))
        self.tools = (*tools, self.search_tool)

    def _definition(self, item: Any) -> Any:
        return (getattr(item, "extras", None) or {}).get("agent_core_definition")

    def _authorized(self, item: Any, context: Any) -> bool:
        extras = getattr(item, "extras", None) or {}
        definition = extras.get("agent_core_definition")
        if definition is not None:
            if context.tool_runtime.execution.is_tool_revoked(item.name):
                return False
            executor = extras.get("agent_core_executor")
            allowed = self._authorization.get(item.name, False)
            if not allowed:
                context.tool_runtime.execution.revoke_tool(item.name)
            return allowed
        # Native argument-specific authorization stays at its execution boundary.
        return item.name in self.metadata

    def _available(self, item: Any, context: Any, *, call_id: str | None = None) -> bool:
        definition = self._definition(item)
        metadata = self.metadata.get(item.name, {})
        maximum = definition.max_calls_per_run if definition else None
        group = definition.budget_group if definition else metadata.get("budget_group", "external")
        return context.tool_runtime.execution.is_available(item.name, maximum, group, call_id=call_id)

    def restore(self, state: Any, context: Any) -> None:
        modes = state.get("tool_visibility", {}).get("modes")
        current_modes = {"discovery": context.settings.tool_discovery_enabled, "retry": context.settings.tool_retry_enabled}
        if modes is not None and modes != current_modes:
            raise AgentContextError("Cannot change tool discovery/retry mode while resuming a Run")
        if not context.settings.tool_discovery_enabled:
            if state.get("tool_discovery"):
                raise AgentContextError("Cannot change tool discovery mode while resuming a Run")
            return
        if self.coordinator is None:
            self.coordinator = DiscoveryCoordinator(ToolCatalog.from_tools(self.tools, self.metadata))
        answered = {message.tool_call_id for message in state.get("messages", ()) if isinstance(message, ToolMessage)}
        pending = [call["name"] for message in state.get("messages", ()) if isinstance(message, AIMessage)
                   for call in message.tool_calls if call["id"] not in answered]
        self.coordinator.restore(state.get("tool_discovery"), pending_tools=pending)

    def _select(self, request: Any, *, finishing: bool) -> tuple[list[Any], str, dict[str, Any]]:
        context = request.runtime.context
        self.restore(request.state, context)
        modes = {"discovery": context.settings.tool_discovery_enabled, "retry": context.settings.tool_retry_enabled}
        if context.settings.tool_discovery_enabled:
            assert self.coordinator is not None
            if finishing:
                return [], "", {"tool_discovery": self.coordinator.snapshot(),
                                "tool_visibility": {"identities": [], "modes": modes}}
            contributed = {item.name for item in request.tools}
            self._plan = self.coordinator.plan(
                authorized=lambda entry: entry.name in contributed and self._authorized(entry.tool, context),
                available=lambda entry: self._available(entry.tool, context),
                schema_budget=(min(context.settings.max_tool_schema_tokens, context.settings.max_input_tokens // 4)
                               if context.settings.max_input_tokens is not None else context.settings.max_tool_schema_tokens),
                required_direct=() if finishing else self.policy.required_direct_tools(context, request.state),
                optional_direct=self.policy.optional_direct_tools(context),
            )
            tools = [] if finishing else list(self._plan.tools)
            visibility = {"identities": [] if finishing else list(self._plan.identities),
                          "schema_budget": self._plan.schema_budget,
                          "deferred_names": list(self._plan.deferred_names), "modes": modes}
            return tools, "" if finishing else self._plan.summary, {
                "tool_discovery": self.coordinator.snapshot(), "tool_visibility": visibility}
        tools = []
        for item in request.tools:
            if item is self.search_tool or item.name == "search_tools" or finishing:
                continue
            definition = self._definition(item)
            discovery = definition.discovery if definition else self.metadata.get(item.name, {}).get("discovery")
            exposure = discovery.get("exposure") if isinstance(discovery, dict) else getattr(discovery, "exposure", None)
            if exposure == "hidden" or not self._available(item, context):
                continue
            if definition is not None and not self._authorized(item, context):
                continue
            tools.append(item)
        return tools, "", {"tool_visibility": {"identities": [{"name": item.name} for item in tools], "modes": modes}}

    def _saved_plan(self, state: Any) -> ExposurePlan:
        if self._plan is not None:
            return self._plan
        assert self.coordinator is not None
        visible = state.get("tool_visibility", {})
        identities = tuple(visible.get("identities", ()))
        names = tuple(item["name"] for item in identities)
        return ExposurePlan(tuple(self.coordinator.catalog.by_name[name].tool for name in names), names,
            identities, 0, visible["schema_budget"], "", tuple(visible.get("deferred_names", ())), True)

    def check_call(self, request: Any) -> tuple[str, str] | None:
        context = request.runtime.context
        self.restore(request.state, context)
        name = request.tool_call["name"]
        if name == "__invalid_tool_call__":
            return "invalid_tool_call", "工具调用格式无效，请按本轮工具参数定义重新生成调用。"
        identities = request.state.get("tool_visibility", {}).get("identities", ())
        visible = {item["name"]: item for item in identities}
        if name not in visible:
            correction = "请先搜索并在下一轮调用。" if "search_tools" in visible else "请使用本轮已绑定的可用工具。"
            return "tool_not_exposed", "工具未在发起调用的模型请求中加载；" + correction
        item = request.tool
        if item is None:
            return "invalid_tool_call", "工具不可用。"
        if self.coordinator is not None:
            entry = self.coordinator.catalog.by_name.get(name)
            if entry is None or entry.identity != visible[name]:
                raise ToolDiscoveryError("tool_catalog_changed", "The requested tool definition changed")
            if not self.coordinator.permitted(name, authorized=lambda candidate: self._authorized(candidate.tool, context)):
                return "tool_not_allowed", "该能力已从本次运行的授权集合移除；新增授权需开启新的运行。"
        if self._definition(item) is not None and not self._authorized(item, context):
            return "tool_not_allowed", "本轮工具权限已失效。"
        if not self._available(item, context, call_id=request.tool_call["id"]):
            return "tool_limit_reached", "该工具或调用分组的额度已耗尽，可使用其他可用能力。"
        return None

    def checkpoint_update(self) -> dict[str, Any]:
        return {"tool_discovery": self.coordinator.snapshot()} if self.coordinator is not None else {}

    def retry_definition(self, name: str) -> Any:
        item = next((item for item in self.tools if item.name == name), None)
        return self._definition(item) if item is not None else None

    async def _refresh_authorization(self, context):
        decisions = {}
        for item in self.tools:
            extras = getattr(item, "extras", None) or {}
            definition = extras.get("agent_core_definition")
            executor = extras.get("agent_core_executor")
            if definition is not None and executor is not None:
                decision = await call_port(executor.authorizer, "authorize", definition, context.tool_runtime)
                decisions[item.name] = decision.allowed
        self._authorization = decisions

    async def aselect(self, request, *, finishing):
        await self._refresh_authorization(request.runtime.context)
        return self._select(request, finishing=finishing)

    def select(self, request, *, finishing):
        return run_sync(self.aselect(request, finishing=finishing))

    async def acheck_call(self, request):
        await self._refresh_authorization(request.runtime.context)
        return self.check_call(request)
