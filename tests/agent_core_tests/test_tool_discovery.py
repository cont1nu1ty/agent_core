"""Discovery algorithms, schema budgets and checkpoint identity contracts."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from agent_core.tools.contracts import ToolDiscoverySpec, ToolSpec
from agent_core.tools.discovery import (
    DiscoveryCoordinator, ToolCatalog, ToolDiscoveryError, merge_discovery_state, tokenize_search_text,
)


class QueryInput(BaseModel):
    query: str = Field(default="", description="查询关键字")


def make_tool(name, description="Read the requested information", *, schema=QueryInput):
    return StructuredTool.from_function(lambda **kwargs: kwargs, name=name,
                                        description=description, args_schema=schema)


def catalog_for(*tools, metadata=None):
    tools = [make_tool("search_tools", "Search the available tool catalog"), *tools]
    values = {tool.name: {"version": "1.0.0", "category": "general",
                         "discovery": {"exposure": "deferred"}} for tool in tools}
    values["search_tools"]["discovery"] = {"exposure": "direct", "required": True}
    for name, entry in (metadata or {}).items():
        values[name].update(entry)
    return ToolCatalog.from_tools(tools, values)


def make_plan(coordinator, **kwargs):
    return coordinator.plan(authorized=lambda _: True, available=lambda _: True,
                            schema_budget=10_000, **kwargs)


def search(coordinator, plan, query, call_id="search-1", **kwargs):
    return coordinator.search(query, plan=plan, call_id=call_id,
                              authorized=lambda _: True, available=lambda _: True, **kwargs)


class FixedRanker:
    """Inject a deliberately bad ranking result without changing catalog policy."""

    def __init__(self, entries=(), *, version="fixed-ranker-1"):
        self.entries = tuple(entries)
        self.version = version
        self.seen_candidates = ()

    def rank(self, _query, candidates, *, limit):
        self.seen_candidates = tuple(entry.name for entry in candidates)
        return list(self.entries)


class CatalogTest(unittest.TestCase):
    def test_catalog_preserves_original_tool_and_uses_explicit_metadata(self):
        tool = make_tool("read_status")
        definition = ToolSpec("read_status", "Read status", QueryInput, lambda *_: None,
                              discovery=ToolDiscoverySpec(aliases=("状态查询",)))
        tool.extras = {"agent_core_definition": definition}
        catalog = ToolCatalog.from_tools([tool])
        self.assertIs(catalog.get("read_status").tool, tool)
        self.assertIs(catalog.get("read_status").definition, definition)
        self.assertEqual(catalog.get("read_status").aliases, ("状态查询",))
        self.assertEqual(catalog.get("read_status").max_calls_per_run, 8)

    def test_hashes_are_registration_order_independent_and_schema_sensitive(self):
        one, two = make_tool("read_one"), make_tool("read_two")
        first = catalog_for(one, two)
        second = catalog_for(two, one)
        self.assertEqual(first.revision, second.revision)
        changed_schema = create_model("ChangedInput", node_id=(str, ...))
        third = catalog_for(make_tool("read_one", schema=changed_schema), two)
        self.assertNotEqual(first.get("read_one").schema_hash, third.get("read_one").schema_hash)
        self.assertNotEqual(first.revision, third.revision)

    def test_unannotated_native_tools_and_duplicate_names_fail_at_assembly(self):
        with self.assertRaises(ToolDiscoveryError) as caught:
            ToolCatalog.from_tools([make_tool("read_one")])
        self.assertEqual(caught.exception.code, "missing_tool_metadata")
        with self.assertRaises(ToolDiscoveryError) as caught:
            catalog_for(make_tool("read_one"), make_tool("read_one"))
        self.assertEqual(caught.exception.code, "duplicate_tool_name")

    def test_normalization_keeps_names_words_chinese_terms_and_bigrams(self):
        tokens = tokenize_search_text("ＧＥＴ_Node_Neighbors 相邻设备遥测走势", ("相邻设备",))
        self.assertIn("get_node_neighbors", tokens)
        self.assertIn("neighbors", tokens)
        self.assertIn("相邻设备", tokens)
        self.assertIn("走势", tokens)

    def test_exact_names_and_aliases_win_with_stable_ties(self):
        catalog = catalog_for(make_tool("read_alpha", "status " * 20), make_tool("read_beta", "status"), metadata={
            "read_beta": {"aliases": ("查运行状态",)},
        })
        candidates = [catalog.get("read_alpha"), catalog.get("read_beta")]
        self.assertEqual(catalog.rank("read_beta", candidates)[0].name, "read_beta")
        self.assertEqual(catalog.rank("查运行状态", candidates)[0].name, "read_beta")
        tied = catalog_for(make_tool("read_alpha", "same"), make_tool("read_beta", "same"))
        self.assertEqual([entry.name for entry in tied.rank("same", [tied.get("read_beta"), tied.get("read_alpha")])],
                         ["read_alpha", "read_beta"])


class DiscoveryStateTest(unittest.TestCase):
    def test_one_search_loads_tools_from_two_intent_clusters(self):
        catalog = catalog_for(
            make_tool("resolve_party", "Resolve an opaque party identifier"),
            make_tool("aggregate_history", "Aggregate historical buckets"),
            metadata={
                "resolve_party": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",),
                    "when_to_use": "Resolve a party identifier from supplied details."}},
                "aggregate_history": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("purchase_trend",),
                    "when_to_use": "Aggregate past orders into time buckets."}},
            },
        )
        coordinator = DiscoveryCoordinator(catalog)
        initial = make_plan(coordinator)
        self.assertEqual(catalog.get("resolve_party").intent_clusters, ("customer_lookup",))
        self.assertEqual(catalog.rank("customer lookup and purchase trend",
                                      [catalog.get("resolve_party"), catalog.get("aggregate_history")]), [])
        receipt = search(coordinator, initial, "customer lookup and purchase trend", limit=2)

        self.assertEqual(receipt["status"], "loaded")
        self.assertEqual({item["name"] for item in receipt["matches"]},
                         {"resolve_party", "aggregate_history"})
        self.assertEqual(initial.names, ("search_tools",))
        self.assertTrue({"resolve_party", "aggregate_history"}.issubset(make_plan(coordinator).names))

    def test_negative_usage_metadata_does_not_create_a_positive_intent_match(self):
        catalog = catalog_for(
            make_tool("describe_schema", "Describe metadata definitions"),
            make_tool("fetch_records", "Read stored row values"),
            metadata={
                "describe_schema": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("schema_inspection",),
                    "when_to_use": "Inspect table schema and columns."}},
                "fetch_records": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("record_read",),
                    "when_to_use": "Retrieve stored records.",
                    "when_not_to_use": "Inspect table schema and columns."}},
            },
        )
        coordinator = DiscoveryCoordinator(catalog)
        receipt = search(coordinator, make_plan(coordinator), "inspect table schema and columns")

        self.assertEqual([item["name"] for item in receipt["matches"]], ["describe_schema"])
        self.assertEqual(set(coordinator.snapshot()["discovered"]), {"describe_schema"})

    def test_intent_cluster_filters_permission_and_hidden_members_before_ranking(self):
        catalog = catalog_for(
            make_tool("resolve_party", "Resolve an opaque party identifier"),
            make_tool("denied_lookup", "Resolve a private opaque identifier"),
            make_tool("secret_lookup", "Resolve a secret opaque identifier"),
            metadata={
                "resolve_party": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",)}},
                "denied_lookup": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",)}},
                "secret_lookup": {"discovery": {"exposure": "hidden",
                    "intent_clusters": ("customer_lookup",)}},
            },
        )
        coordinator = DiscoveryCoordinator(catalog)
        authorized = lambda entry: entry.name != "denied_lookup"
        plan = coordinator.plan(authorized=authorized, available=lambda _: True, schema_budget=10_000)
        receipt = coordinator.search("customer lookup", call_id="one", plan=plan,
                                     authorized=authorized, available=lambda _: True, limit=8)

        self.assertEqual([item["name"] for item in receipt["matches"]], ["resolve_party"])
        visible = json.dumps([receipt, plan.summary, coordinator.snapshot()], ensure_ascii=False)
        self.assertNotIn("denied_lookup", visible)
        self.assertNotIn("secret_lookup", visible)

    def test_hidden_or_denied_alias_cannot_change_authorized_default_ranking(self):
        def authorized_results(extra_exposure=None):
            tools = [make_tool("read_history", "客户订单 历史"),
                     make_tool("read_status", "订单状态")]
            metadata = {}
            if extra_exposure is not None:
                tools.append(make_tool("private_lookup", "其他能力"))
                metadata["private_lookup"] = {"discovery": {
                    "exposure": extra_exposure, "aliases": ("客户订单",)}}
            coordinator = DiscoveryCoordinator(catalog_for(*tools, metadata=metadata))
            authorized = lambda entry: entry.name != "private_lookup"
            plan = coordinator.plan(authorized=authorized, available=lambda _: True, schema_budget=10_000)
            receipt = coordinator.search("客户订单状态", call_id="one", plan=plan,
                                         authorized=authorized, available=lambda _: True, limit=2)
            return [item["name"] for item in receipt["matches"]]

        # The private alias would otherwise change Chinese tokenization and
        # reverse these two authorized results, even though it is ineligible.
        baseline = authorized_results()
        self.assertEqual(baseline, ["read_status", "read_history"])
        for exposure in ("hidden", "deferred"):
            with self.subTest(exposure=exposure):
                self.assertEqual(authorized_results(exposure), baseline)

    def test_explicitly_negated_intent_does_not_load_its_tool(self):
        catalog = catalog_for(
            make_tool("read_series", "Read historical measurements"),
            make_tool("read_snapshot", "Read recent aggregate values"),
            metadata={
                "read_series": {"discovery": {"exposure": "deferred",
                    "aliases": ("量测趋势", "历史走势"),
                    "intent_clusters": ("measurement_trend",),
                    "when_to_use": "读取单指标量测历史趋势"}},
                "read_snapshot": {"discovery": {"exposure": "deferred",
                    "aliases": ("多指标摘要", "近期摘要"),
                    "intent_clusters": ("measurement_summary",),
                    "when_to_use": "读取最近多指标状态摘要"}},
            },
        )
        coordinator = DiscoveryCoordinator(catalog)
        receipt = search(coordinator, make_plan(coordinator), "不要量测趋势，只看最近多指标摘要", limit=2)

        self.assertEqual([item["name"] for item in receipt["matches"]], ["read_snapshot"])
        self.assertEqual(set(coordinator.snapshot()["discovered"]), {"read_snapshot"})
        self.assertNotIn("read_series", make_plan(coordinator).names)

    def test_bad_ranker_cannot_commit_ineligible_or_duplicate_discoveries(self):
        catalog = catalog_for(
            make_tool("allowed_lookup", "Find an allowed record"),
            make_tool("denied_lookup", "Find a denied record"),
            make_tool("hidden_lookup", "Find a hidden record"),
            metadata={"hidden_lookup": {"discovery": {"exposure": "hidden"}}},
        )
        for returned in (("allowed_lookup", "denied_lookup"),
                         ("allowed_lookup", "hidden_lookup"),
                         ("allowed_lookup", "allowed_lookup")):
            with self.subTest(returned=returned):
                ranker = FixedRanker([catalog.get(name) for name in returned])
                coordinator = DiscoveryCoordinator(catalog, ranker=ranker)
                authorized = lambda entry: entry.name != "denied_lookup"
                plan = coordinator.plan(authorized=authorized, available=lambda _: True, schema_budget=10_000)
                before = coordinator.snapshot()
                with self.assertRaises(ToolDiscoveryError) as caught:
                    coordinator.search("record lookup", call_id="bad", plan=plan,
                                       authorized=authorized, available=lambda _: True, limit=2)
                self.assertEqual(caught.exception.code, "invalid_tool_ranker")
                self.assertEqual(ranker.seen_candidates, ("allowed_lookup",))
                after = coordinator.snapshot()
                self.assertEqual(after["discovered"], {})
                self.assertEqual(after["search_ledger"], {})
                self.assertEqual(after["revision"], before["revision"])

    def test_intent_cluster_batch_only_commits_tools_that_fit_schema_budget(self):
        catalog = catalog_for(
            make_tool("resolve_party", "Resolve an opaque party identifier"),
            make_tool("aggregate_history", "Aggregate historical buckets"),
            metadata={
                "resolve_party": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",)}},
                "aggregate_history": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("purchase_trend",)}},
            },
        )
        budget = count_tokens_approximately([], tools=[catalog.get("search_tools").tool,
                                                        catalog.get("resolve_party").tool])
        coordinator = DiscoveryCoordinator(catalog)
        plan = coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=budget)
        receipt = search(coordinator, plan, "customer lookup and purchase trend", limit=2)

        self.assertEqual(receipt["status"], "loaded")
        self.assertEqual(len(receipt["matches"]), 1)
        self.assertEqual(receipt["omitted_for_budget"], 1)
        self.assertEqual(set(coordinator.snapshot()["discovered"]),
                         {item["name"] for item in receipt["matches"]})
        rebound = coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=budget)
        self.assertLessEqual(rebound.schema_tokens, budget)
        self.assertEqual(set(rebound.names) - {"search_tools"}, set(coordinator.snapshot()["discovered"]))

    def test_exact_name_precedes_cluster_neighbors_and_unrelated_query_is_empty(self):
        catalog = catalog_for(
            make_tool("read_profile", "Read a customer profile"),
            make_tool("resolve_party", "Resolve a customer identifier"),
            metadata={
                "read_profile": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",)}},
                "resolve_party": {"discovery": {"exposure": "deferred",
                    "intent_clusters": ("customer_lookup",)}},
            },
        )
        coordinator = DiscoveryCoordinator(catalog)
        plan = make_plan(coordinator)
        exact = search(coordinator, plan, "resolve_party", "exact", limit=2)
        unrelated = search(coordinator, plan, "quantum zebra cinnamon", "outside")

        self.assertEqual(exact["matches"][0]["name"], "resolve_party")
        self.assertEqual(unrelated["status"], "no_match")
        self.assertEqual(unrelated["matches"], [])

    def test_search_updates_next_plan_but_not_originating_exposure_snapshot(self):
        coordinator = DiscoveryCoordinator(catalog_for(make_tool("read_status", "设备状态")))
        initial = make_plan(coordinator)
        result = search(coordinator, initial, "设备状态")
        self.assertEqual(result["status"], "loaded")
        self.assertEqual(initial.names, ("search_tools",))
        self.assertEqual(make_plan(coordinator).names, ("read_status", "search_tools"))
        self.assertNotIn("schema", result["matches"][0])

    def test_two_searches_on_the_same_old_plan_accumulate_and_respect_budget(self):
        catalog = catalog_for(make_tool("read_alpha", "alpha"), make_tool("read_beta", "beta"))
        coordinator = DiscoveryCoordinator(catalog)
        budget = count_tokens_approximately([], tools=[catalog.get("search_tools").tool, catalog.get("read_alpha").tool])
        plan = coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=budget)
        first = search(coordinator, plan, "read_alpha", "one", limit=1)
        saved = coordinator.snapshot()
        second = search(coordinator, plan, "read_beta", "two", limit=1)
        self.assertEqual(first["status"], "loaded")
        self.assertEqual(second["status"], "tool_schema_budget_exhausted")
        coordinator.restore(saved)
        self.assertEqual(set(coordinator.snapshot()["search_ledger"]), {"one", "two"})
        self.assertEqual(set(coordinator.snapshot()["discovered"]), {"read_alpha"})

    def test_repeated_match_does_not_allocate_schema_twice(self):
        coordinator = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态")))
        plan = make_plan(coordinator)
        search(coordinator, plan, "状态", "one")
        receipt = search(coordinator, plan, "状态", "two")
        self.assertTrue(receipt["matches"][0]["already_loaded"])
        self.assertEqual(len(coordinator.snapshot()["discovered"]), 1)
        with self.assertRaises(ToolDiscoveryError) as caught:
            search(coordinator, plan, "状态", "two")
        self.assertEqual(caught.exception.code, "duplicate_tool_call")

    def test_search_receipt_is_compact_but_actual_schema_keeps_full_description(self):
        description = "检索已授权文件。" * 100
        tool = make_tool("read_long", description)
        coordinator = DiscoveryCoordinator(catalog_for(tool))
        result = search(coordinator, make_plan(coordinator), "read_long")
        self.assertLessEqual(len(result["matches"][0]["description"]), 320)
        self.assertNotIn(description, json.dumps(result, ensure_ascii=False))
        exposed = next(item for item in make_plan(coordinator).tools if item.name == "read_long")
        self.assertEqual(exposed.description, description)

    def test_filtering_precedes_top_k_and_hidden_names_do_not_leak(self):
        tools = [make_tool(f"denied_{index}", "设备状态 " * 15) for index in range(12)]
        tools += [make_tool("secret_audit", "绝密设备状态"), make_tool("read_allowed", "设备状态")]
        catalog = catalog_for(*tools, metadata={"secret_audit": {"category": "classified",
                                     "discovery": {"exposure": "hidden"}}})
        coordinator = DiscoveryCoordinator(catalog)
        authorized = lambda entry: not entry.name.startswith("denied_")
        plan = coordinator.plan(authorized=authorized, available=lambda _: True, schema_budget=10_000)
        result = coordinator.search("设备状态", call_id="one", plan=plan,
                                    authorized=authorized, available=lambda _: True, limit=1)
        self.assertEqual([item["name"] for item in result["matches"]], ["read_allowed"])
        projection = json.dumps([result, plan.summary], ensure_ascii=False)
        self.assertNotIn("secret_audit", projection)
        self.assertNotIn("classified", projection)
        self.assertNotIn("denied_", projection)

    def test_authorization_only_tightens_within_a_run(self):
        catalog = catalog_for(make_tool("read_status", "状态"), make_tool("read_other", "其他"))
        coordinator = DiscoveryCoordinator(catalog)
        plan = make_plan(coordinator)
        search(coordinator, plan, "状态", "one")
        narrow = coordinator.plan(authorized=lambda entry: entry.name != "read_status", available=lambda _: True,
                                  schema_budget=10_000)
        self.assertNotIn("read_status", narrow.names)
        self.assertNotIn("read_status", make_plan(coordinator).names)
        restored = DiscoveryCoordinator(catalog)
        restored.restore(coordinator.snapshot())
        self.assertNotIn("read_status", make_plan(restored).names)

    def test_execution_observed_revocation_cannot_be_regranted_by_old_visibility(self):
        catalog = catalog_for(make_tool("read_status", "状态"))
        coordinator = DiscoveryCoordinator(catalog)
        original = make_plan(coordinator, required_direct=("read_status",))
        self.assertIn("read_status", original.names)
        self.assertFalse(coordinator.permitted("read_status", authorized=lambda entry: entry.name != "read_status"))
        self.assertFalse(coordinator.permitted("read_status", authorized=lambda _: True))
        fresh = DiscoveryCoordinator(catalog)
        fresh.restore(coordinator.snapshot())
        self.assertFalse(fresh.permitted("read_status", authorized=lambda _: True))
        self.assertTrue(fresh.permitted("search_tools", authorized=lambda _: True))

    def test_unavailable_category_and_bad_queries_have_safe_codes(self):
        coordinator = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态")))
        plan = make_plan(coordinator)
        for kwargs in ({"query": ""}, {"query": "x" * 513}, {"query": "状态", "limit": 9},
                       {"query": "状态", "limit": True}, {"query": "状态", "category": "secret"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ToolDiscoveryError):
                search(coordinator, plan, **kwargs)
        receipt = search(coordinator, plan, "quuxvexxy")
        self.assertEqual(receipt["status"], "no_match")
        self.assertEqual(receipt["matches"], [])
        self.assertNotIn("quuxvexxy", json.dumps(coordinator.snapshot()))

    def test_optional_direct_demotion_and_required_schema_failure(self):
        catalog = catalog_for(make_tool("read_status", "状态"), make_tool("read_other", "其他"), metadata={
            "read_status": {"discovery": {"exposure": "direct", "direct_priority": 20}},
        })
        coordinator = DiscoveryCoordinator(catalog)
        budget = catalog.get("search_tools").schema_tokens
        plan = coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=budget)
        self.assertEqual(plan.names, ("search_tools",))
        self.assertIn("read_status", plan.deferred_names)
        with self.assertRaises(ToolDiscoveryError) as caught:
            coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=budget,
                             required_direct=("read_status",))
        self.assertEqual(caught.exception.code, "tool_schema_budget_exhausted")

    def test_legacy_full_exposure_has_no_search_and_still_checks_schema_budget(self):
        catalog = catalog_for(make_tool("read_status", "状态"))
        coordinator = DiscoveryCoordinator(catalog)
        plan = make_plan(coordinator, discovery_enabled=False)
        self.assertEqual(plan.names, ("read_status",))
        with self.assertRaises(ToolDiscoveryError):
            coordinator.plan(authorized=lambda _: True, available=lambda _: True, schema_budget=1, discovery_enabled=False)

    def test_all_optional_direct_tools_can_be_demoted_without_losing_search(self):
        catalog = catalog_for(make_tool("read_long", "工具说明" * 100), metadata={
            "read_long": {"discovery": {"exposure": "direct"}},
        })
        plan = DiscoveryCoordinator(catalog).plan(authorized=lambda _: True, available=lambda _: True,
            schema_budget=catalog.get("search_tools").schema_tokens)
        self.assertEqual(plan.names, ("search_tools",))
        self.assertEqual(plan.deferred_names, ("read_long",))

    def test_exhausted_search_is_not_promoted_by_the_capability_summary(self):
        coordinator = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态")))
        plan = coordinator.plan(authorized=lambda _: True, available=lambda entry: entry.name != "search_tools",
                                schema_budget=10_000)
        self.assertNotIn("search_tools", plan.names)
        self.assertNotIn("search_tools", plan.summary)
        self.assertIn("发现不可用", plan.summary)

    def test_restore_retains_discoveries_but_ignores_new_catalog_entries(self):
        old = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态")))
        search(old, make_plan(old), "状态")
        restored = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态"), make_tool("read_new", "新能力")))
        restored.restore(old.snapshot())
        plan = make_plan(restored)
        self.assertIn("read_status", plan.names)
        self.assertNotIn("read_new", plan.deferred_names)
        self.assertEqual(len(restored.snapshot()["search_ledger"]), 1)

    def test_restore_rejects_ranker_or_trusted_intent_change_but_allows_new_tool(self):
        baseline_metadata = {"read_record": {"discovery": {"exposure": "deferred",
            "intent_clusters": ("record_lookup",), "when_to_use": "Locate a stored record."}}}
        original = catalog_for(make_tool("read_record", "Read a record"), metadata=baseline_metadata)
        initial = DiscoveryCoordinator(original, ranker=FixedRanker(version="stable-1"))
        make_plan(initial)
        saved = initial.snapshot()

        with self.assertRaises(ToolDiscoveryError) as caught:
            DiscoveryCoordinator(original, ranker=FixedRanker(version="stable-2")).restore(saved)
        self.assertEqual(caught.exception.code, "tool_catalog_changed")
        for changed_discovery in (
            {"exposure": "deferred", "intent_clusters": ("archive_lookup",),
             "when_to_use": "Locate a stored record."},
            {"exposure": "deferred", "intent_clusters": ("record_lookup",),
             "when_to_use": "Locate an archived record."},
        ):
            with self.subTest(changed_discovery=changed_discovery):
                changed = catalog_for(make_tool("read_record", "Read a record"),
                                      metadata={"read_record": {"discovery": changed_discovery}})
                with self.assertRaises(ToolDiscoveryError) as caught:
                    DiscoveryCoordinator(changed, ranker=FixedRanker(version="stable-1")).restore(saved)
                self.assertEqual(caught.exception.code, "tool_catalog_changed")

        extended = catalog_for(make_tool("read_record", "Read a record"),
                               make_tool("read_new", "Read a newly added record"),
                               metadata=baseline_metadata)
        restored = DiscoveryCoordinator(extended, ranker=FixedRanker(version="stable-1"))
        restored.restore(saved)
        plan = make_plan(restored)
        self.assertIn("read_record", plan.deferred_names)
        self.assertNotIn("read_new", plan.deferred_names)

    def test_exposed_discovered_and_pending_schema_changes_are_rejected(self):
        original = catalog_for(make_tool("read_status", "状态"))
        initial = DiscoveryCoordinator(original)
        make_plan(initial)
        changed = catalog_for(make_tool("read_status", "状态", schema=create_model("NewInput", other=(int, ...))))
        unobserved = DiscoveryCoordinator(changed)
        unobserved.restore(initial.snapshot())
        self.assertNotIn("read_status", make_plan(unobserved).deferred_names)
        with self.assertRaises(ToolDiscoveryError) as caught:
            DiscoveryCoordinator(changed).restore(initial.snapshot(), pending_tools=("read_status",))
        self.assertEqual(caught.exception.code, "tool_catalog_changed")
        search(initial, make_plan(initial), "状态")
        with self.assertRaises(ToolDiscoveryError):
            DiscoveryCoordinator(changed).restore(initial.snapshot())

    def test_snapshot_is_detached_and_revision_reducer_is_monotonic(self):
        coordinator = DiscoveryCoordinator(catalog_for(make_tool("read_status", "状态")))
        search(coordinator, make_plan(coordinator), "状态")
        first = coordinator.snapshot()
        corrupted = coordinator.snapshot()
        corrupted["discovered"].clear()
        self.assertTrue(coordinator.snapshot()["discovered"])
        old = {**first, "revision": first["revision"] - 1}
        self.assertIs(merge_discovery_state(first, old), first)
        self.assertIs(merge_discovery_state(old, first), first)




if __name__ == "__main__":
    unittest.main()
