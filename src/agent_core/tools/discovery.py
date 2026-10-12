"""Deterministic tool discovery over trusted metadata and existing tool objects.

The catalog is a projection, not an execution registry. Its coordinator mirrors
checkpoint state within one Run so serial tool tasks sharing an older graph
snapshot cannot overwrite discoveries or spend the same schema space twice.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import re
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol
import unicodedata

from langchain_core.messages.utils import count_tokens_approximately
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_core.contracts.errors import AgentContextError


class ToolDiscoveryError(AgentContextError):
    """Safe discovery failure with a stable code for the caller's ToolMessage."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.public_message = message
        super().__init__(message)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    raise ToolDiscoveryError("invalid_tool_catalog", "Tool metadata must be a trusted mapping")


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip().lower()


_INTENT_CLAUSE_SPLIT = re.compile(r"[，,;；。!?！？]|\b(?:but|instead|rather than)\b|(?:但是|但|而是|只看|只要|改为)")
_NEGATED_INTENT = re.compile(
    r"^(?:(?:请|please)\s*)?(?:不要|不用|无需|无须|不需要|别用|别查|别看|排除|不是|"
    r"do\s+not|don't|dont|without|exclude|not(?!\s+only))\s*"
)


def _explicit_intent_clauses(query: str) -> tuple[str, tuple[str, ...]]:
    """Keep affirmative requests separate from explicit exclusion clauses."""
    positive: list[str] = []
    negative: list[str] = []
    for clause in _INTENT_CLAUSE_SPLIT.split(_normalize(query)):
        clause = clause.strip()
        if not clause:
            continue
        match = _NEGATED_INTENT.match(clause)
        if match:
            negative.append(clause[match.end():].strip())
        else:
            positive.append(clause)
    return " ".join(positive), tuple(part for part in negative if part)


def _mentions_phrase(text: str, phrase: str) -> bool:
    phrase = _normalize(phrase)
    if phrase.isascii():
        return bool(re.search(rf"(?<![a-z0-9_]){re.escape(phrase)}(?![a-z0-9_])", text))
    return phrase in text


_DOMAIN_TERMS = (
    "工具", "设备", "节点", "连接", "拓扑", "上下游", "上游", "下游", "邻接", "一跳", "供电", "路径",
    "量测", "遥测", "时序", "趋势", "负载", "健康", "规程", "知识", "隔离", "边界", "连通", "孤岛",
    "转供", "结构", "关键", "质量", "记忆", "历史", "来源", "笔记", "计划", "文件", "读取", "搜索",
)
_CJK = re.compile(r"[\u3400-\u9fff]+")
_ENGLISH_STOPWORDS = frozenset("a an the and or for with of in on to by from is are be as at using use "
    "get read find show please this that these those it its can only then already have has do does not no into".split())


def _word_roots(word: str) -> tuple[str, ...]:
    """Conservative English inflections, keeping the original spelling too."""
    if not word.isascii() or not word.isalpha() or len(word) < 5:
        return ()
    if word.endswith("ies"):
        return (word[:-3] + "y",)
    if word.endswith(("ches", "shes", "xes", "sses", "zes")):
        return (word[:-2],)
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return (word[:-1],)
    if word.endswith("ing") and len(word) > 6:
        return (word[:-3],)
    if word.endswith("ed") and len(word) > 5:
        return (word[:-2],)
    return ()


def tokenize_search_text(text: str, vocabulary: Sequence[str] = ()) -> tuple[str, ...]:
    """Version 1: NFKC, identifier words, controlled Chinese terms and bigrams."""
    normalized = _normalize(text)
    tokens = re.findall(r"[a-z0-9]+(?:_[a-z0-9]+)*", normalized)
    tokens.extend(part for token in tuple(tokens) if "_" in token for part in token.split("_"))
    tokens = [token for token in tokens if token not in _ENGLISH_STOPWORDS]
    tokens.extend(root for token in tuple(tokens) for root in _word_roots(token))
    terms = sorted({*_DOMAIN_TERMS, *(word for word in vocabulary if _CJK.fullmatch(word))},
                   key=lambda item: (-len(item), item))
    for match in _CJK.finditer(normalized):
        run = match.group()
        tokens.extend(run[index:index + 2] for index in range(len(run) - 1))
        if len(run) == 1:
            tokens.append(run)
        index = 0
        while index < len(run):
            word = next((word for word in terms if run.startswith(word, index)), None)
            if word:
                tokens.append(word)
                index += len(word)
            else:
                index += 1
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class ToolCatalogEntry:
    tool: BaseTool
    name: str
    version: str
    schema_hash: str
    description: str
    category: str
    tags: tuple[str, ...]
    aliases: tuple[str, ...]
    intent_clusters: tuple[str, ...]
    when_to_use: str
    when_not_to_use: str
    exposure: str
    direct_priority: int
    required: bool
    source: str
    budget_group: str
    effect: str
    schema_tokens: int

    @property
    def tool_id(self) -> str:
        return self.name

    @property
    def definition(self) -> Any:
        return (self.tool.extras or {}).get("agent_core_definition")

    @property
    def max_calls_per_run(self) -> int | None:
        return getattr(self.definition, "max_calls_per_run", None)

    @property
    def identity(self) -> dict[str, str]:
        return {"name": self.name, "version": self.version, "schema_hash": self.schema_hash}

    @property
    def search_text(self) -> str:
        return " ".join((self.positive_search_text, self.when_not_to_use))

    @property
    def positive_search_text(self) -> str:
        function = convert_to_openai_tool(self.tool)["function"]
        # Runtime injection and credentials are absent from tool_call_schema.
        def properties(schema: Any) -> list[str]:
            if not isinstance(schema, Mapping):
                return []
            result = []
            for key, value in schema.items():
                if key == "properties" and isinstance(value, Mapping):
                    result.extend(value)
                elif key == "description" and isinstance(value, str):
                    result.append(value)
                if isinstance(value, Mapping):
                    result.extend(properties(value))
                elif isinstance(value, list):
                    for item in value:
                        result.extend(properties(item))
            return result
        return " ".join((self.name, self.description, self.category, *self.tags, *self.aliases,
                         self.when_to_use, *properties(function.get("parameters", {}))))

    @property
    def ranking_hash(self) -> str:
        return _hash({"name": self.name, "description": self.description, "category": self.category,
                      "tags": self.tags, "aliases": self.aliases, "when_to_use": self.when_to_use,
                      "when_not_to_use": self.when_not_to_use, "intents": self.intent_clusters})


class ToolCatalog:
    """Immutable metadata projection retaining each existing BaseTool object."""

    def __init__(self, entries: Sequence[ToolCatalogEntry]) -> None:
        ordered = tuple(sorted(entries, key=lambda item: item.name))
        names = [entry.name for entry in ordered]
        if len(names) != len(set(names)):
            raise ToolDiscoveryError("duplicate_tool_name", "Tool names must be unique across all contributors")
        self.entries = ordered
        self.by_name = MappingProxyType({entry.name: entry for entry in ordered})
        self.revision = _hash([{
            **entry.identity, "description": entry.description, "category": entry.category,
            "tags": entry.tags, "aliases": entry.aliases, "intent_clusters": entry.intent_clusters,
            "when_to_use": entry.when_to_use,
            "when_not_to_use": entry.when_not_to_use, "exposure": entry.exposure,
            "direct_priority": entry.direct_priority, "required": entry.required,
            "source": entry.source, "budget_group": entry.budget_group, "effect": entry.effect,
        } for entry in ordered])
        self.vocabulary = tuple(sorted({alias for entry in ordered for alias in entry.aliases}))
        self._tokens = {entry.name: Counter(tokenize_search_text(entry.search_text, self.vocabulary))
                        for entry in ordered}
        self._positive_texts = {entry.name: entry.positive_search_text for entry in ordered}

    def get(self, name: str) -> ToolCatalogEntry:
        return self.by_name[name]

    @classmethod
    def from_tools(cls, tools: Sequence[BaseTool],
                   metadata: Mapping[str, Mapping[str, Any]] | None = None) -> ToolCatalog:
        contributions = metadata or {}
        entries = []
        for tool in tools:
            if not isinstance(tool, BaseTool):
                raise ToolDiscoveryError("invalid_tool_catalog", "Discovery requires registered BaseTool instances")
            extras = tool.extras or {}
            definition = extras.get("agent_core_definition")
            values: dict[str, Any] = {}
            if definition is not None:
                values = {name: getattr(definition, name) for name in (
                    "version", "category", "tags", "discovery", "source", "budget_group", "effect")}
            if "agent_core_catalog" in extras:
                values.update(_mapping(extras["agent_core_catalog"]))
            if tool.name in contributions:
                values.update(_mapping(contributions[tool.name]))
            if not values:
                raise ToolDiscoveryError("missing_tool_metadata", "Every discovery tool requires trusted catalog metadata")
            discovery = _mapping(values.get("discovery", {}))
            exposure = discovery.get("exposure", values.get("exposure", "deferred"))
            if exposure not in {"direct", "deferred", "hidden"}:
                raise ToolDiscoveryError("invalid_tool_catalog", "Tool exposure must be direct, deferred or hidden")
            aliases = discovery.get("aliases", values.get("aliases", ()))
            if isinstance(aliases, str) or not all(isinstance(alias, str) and alias.strip() for alias in aliases):
                raise ToolDiscoveryError("invalid_tool_catalog", "Tool aliases must be nonempty strings")
            clusters = discovery.get("intent_clusters", values.get("intent_clusters", ()))
            if (isinstance(clusters, str) or not isinstance(clusters, (tuple, list))
                    or any(not isinstance(cluster, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", cluster)
                           for cluster in clusters) or len(set(clusters)) != len(clusters)):
                raise ToolDiscoveryError("invalid_tool_catalog", "Intent clusters must be unique snake_case identifiers")
            function = convert_to_openai_tool(tool)["function"]
            entries.append(ToolCatalogEntry(
                tool=tool, name=tool.name, version=str(values.get("version", "1.0.0")),
                schema_hash=_hash(function.get("parameters", {})),
                description=str(values.get("description", tool.description)),
                category=str(values.get("category", "general")),
                tags=tuple(sorted(values.get("tags", ()))), aliases=tuple(sorted(set(aliases))),
                intent_clusters=tuple(sorted(clusters)),
                when_to_use=str(discovery.get("when_to_use", values.get("when_to_use", ""))),
                when_not_to_use=str(discovery.get("when_not_to_use", values.get("when_not_to_use", ""))),
                exposure=exposure, direct_priority=int(discovery.get("direct_priority", values.get("direct_priority", 0))),
                required=bool(discovery.get("required", values.get("required", False))),
                source=str(values.get("source", "native")), budget_group=str(values.get("budget_group", "external")),
                effect=str(values.get("effect", "unknown")),
                schema_tokens=count_tokens_approximately([], tools=[tool]),
            ))
        return cls(entries)

    def rank(self, query: str, candidates: Sequence[ToolCatalogEntry]) -> list[ToolCatalogEntry]:
        """BM25 statistics and Top-K candidates are both permission-filtered."""
        if not candidates:
            return []
        terms = set(tokenize_search_text(query, self.vocabulary))
        normalized = _normalize(query)
        query_identifiers = set(re.findall(r"[a-z][a-z0-9_]*", normalized))
        documents = [self._tokens[entry.name] for entry in candidates]
        lengths = [sum(document.values()) for document in documents]
        average = sum(lengths) / len(lengths) or 1.0
        frequencies = Counter(term for document in documents for term in terms if term in document)
        scored = []
        for entry, document, length in zip(candidates, documents, lengths, strict=True):
            exact_name = normalized == entry.name or entry.name in query_identifiers
            exact_alias = normalized in {_normalize(alias) for alias in entry.aliases}
            score = 0.0
            for term in terms.intersection(document):
                count = document[term]
                inverse = math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                score += inverse * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / average))
            if exact_name or exact_alias or score > 0:
                scored.append((not exact_name, not exact_alias, -score, entry.name, entry))
        return [item[-1] for item in sorted(scored, key=lambda item: item[:-1])]


class ToolRanker(Protocol):
    """Rank only a caller-supplied, already-authorized candidate view."""

    version: str

    def rank(self, query: str, candidates: Sequence[ToolCatalogEntry], *, limit: int) -> list[ToolCatalogEntry]: ...


class IntentClusterRanker:
    """Deterministic, local intent retrieval over trusted multi-label metadata.

    Cluster labels are curated by tool owners. Positive tool descriptions and
    aliases supply bilingual evidence; exclusion prose is never a positive hit.
    No model, remote encoder, or cross-Run index is constructed.
    """

    version = "intent-clusters-1"

    def __init__(self, catalog: ToolCatalog) -> None:
        self.catalog = catalog
        self._candidate_cache: dict[tuple[str, ...], tuple[tuple[str, ...], tuple[Counter[str], ...]]] = {}

    def rank(self, query: str, candidates: Sequence[ToolCatalogEntry], *, limit: int) -> list[ToolCatalogEntry]:
        if not candidates:
            return []
        positive_query, negative_clauses = _explicit_intent_clauses(query)
        if not positive_query:
            return []
        candidates = [entry for entry in candidates if not any(
            _mentions_phrase(clause, phrase)
            for clause in negative_clauses
            for phrase in (entry.name, *entry.aliases,
                           *(cluster.replace("_", " ") for cluster in entry.intent_clusters))
        )]
        if not candidates:
            return []
        # The vocabulary is derived from this authorized candidate view only.
        # Hidden or denied aliases cannot change an allowed tool's tokenization.
        candidate_key = tuple(entry.name for entry in candidates)
        cached = self._candidate_cache.get(candidate_key)
        if cached is None:
            vocabulary = tuple(sorted({alias for entry in candidates for alias in entry.aliases}))
            documents = tuple(Counter(tokenize_search_text(self.catalog._positive_texts[entry.name], vocabulary))
                              for entry in candidates)
            if len(self._candidate_cache) >= 16:
                self._candidate_cache.pop(next(iter(self._candidate_cache)))
            self._candidate_cache[candidate_key] = (vocabulary, documents)
        else:
            vocabulary, documents = cached
        terms = set(tokenize_search_text(positive_query, vocabulary))
        normalized = _normalize(positive_query)
        query_identifiers = set(re.findall(r"[a-z][a-z0-9_]*", normalized))
        lengths = [sum(document.values()) for document in documents]
        average = sum(lengths) / len(lengths) or 1.0
        frequencies = Counter(term for document in documents for term in terms if term in document)

        scored: dict[str, tuple[bool, bool, float, ToolCatalogEntry]] = {}
        cluster_members: dict[str, list[ToolCatalogEntry]] = {}
        for entry, document, length in zip(candidates, documents, lengths, strict=True):
            exact_name = normalized == entry.name or entry.name in query_identifiers
            exact_alias = normalized in {_normalize(alias) for alias in entry.aliases}
            lexical = 0.0
            for term in terms.intersection(document):
                count = document[term]
                inverse = math.log(1 + (len(documents) - frequencies[term] + 0.5) / (frequencies[term] + 0.5))
                lexical += inverse * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * length / average))
            label_score = 0.0
            for cluster in entry.intent_clusters:
                label_terms = set(tokenize_search_text(cluster.replace("_", " ")))
                if label_terms:
                    label_score = max(label_score, 2.0 * len(terms & label_terms) / len(label_terms))
                cluster_members.setdefault(cluster, []).append(entry)
            score = lexical + label_score
            if exact_name or exact_alias or score > 0:
                scored[entry.name] = (exact_name, exact_alias, score, entry)

        def order(item: tuple[bool, bool, float, ToolCatalogEntry]) -> tuple[bool, bool, float, str]:
            exact_name, exact_alias, score, entry = item
            return (not exact_name, not exact_alias, -score, entry.name)

        ordered = sorted(scored.values(), key=order)
        selected: list[ToolCatalogEntry] = []
        covered: set[str] = set()

        def accept(entry: ToolCatalogEntry) -> None:
            selected.append(entry)
            covered.update(entry.intent_clusters)

        for exact_name, exact_alias, _score, entry in ordered:
            if len(selected) >= limit:
                break
            if exact_name or exact_alias:
                accept(entry)
        cluster_scores = {cluster: max((scored[entry.name][2] for entry in members if entry.name in scored), default=0.0)
                          for cluster, members in cluster_members.items()}
        def alias_mentioned(alias: str) -> bool:
            return _mentions_phrase(normalized, alias)

        # A cluster needs independent, owner-curated intent evidence. Ordinary
        # description words alone cannot promote unrelated clusters over a
        # precise memory/file match from another tool owner.
        def cluster_mentioned(cluster: str) -> bool:
            label_terms = set(tokenize_search_text(cluster.replace("_", " ")))
            return (bool(label_terms) and label_terms <= terms) or any(
                alias_mentioned(alias) for entry in cluster_members[cluster] for alias in entry.aliases)

        clusters = sorted((cluster for cluster, score in cluster_scores.items()
                           if score > 0 and cluster_mentioned(cluster)),
                          key=lambda cluster: (-cluster_scores[cluster], cluster))
        diversified = 0
        for cluster in clusters:
            if len(selected) >= limit or diversified >= 3:
                break
            if cluster in covered:
                continue
            members = sorted((scored[entry.name] for entry in cluster_members[cluster]
                              if entry.name in scored and entry not in selected), key=order)
            if members:
                accept(members[0][3])
                diversified += 1
        for _exact_name, _exact_alias, _score, entry in ordered:
            if len(selected) >= limit:
                break
            if entry not in selected:
                accept(entry)
        return selected


@dataclass(frozen=True, slots=True)
class ExposurePlan:
    tools: tuple[BaseTool, ...]
    names: tuple[str, ...]
    identities: tuple[dict[str, str], ...]
    schema_tokens: int
    schema_budget: int
    summary: str
    deferred_names: tuple[str, ...]
    discovery_enabled: bool


def merge_discovery_state(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Serial tasks emit cumulative snapshots; preserve the newest revision."""
    return right if right.get("revision", 0) >= left.get("revision", 0) else left


class DiscoveryCoordinator:
    """One Run's bounded discovery mirror; persistence remains in graph state."""

    def __init__(self, catalog: ToolCatalog, *, ranker: ToolRanker | None = None) -> None:
        self.catalog = catalog
        self.ranker = ranker or IntentClusterRanker(catalog)
        self._lock = RLock()
        self._revision = 0
        self._catalog_revision = catalog.revision
        self._initial_catalog: dict[str, dict[str, str]] | None = None
        self._authorized_names: set[str] = set()
        self._discovered: dict[str, dict[str, str]] = {}
        self._visible_seen: dict[str, dict[str, str]] = {}
        self._ledger: dict[str, dict[str, Any]] = {}

    def _compatible(self, identity: Mapping[str, Any]) -> bool:
        entry = self.catalog.by_name.get(identity.get("name"))
        return entry is not None and entry.identity == dict(identity)

    def _eligible(self, entry: ToolCatalogEntry, authorized: Callable[[ToolCatalogEntry], bool],
                  available: Callable[[ToolCatalogEntry], bool]) -> bool:
        identity = (self._initial_catalog or {}).get(entry.name)
        return (identity is not None and entry.name in self._authorized_names
                and self._compatible(identity) and entry.exposure != "hidden"
                and authorized(entry) and available(entry))

    def _tighten(self, authorized: Callable[[ToolCatalogEntry], bool]) -> None:
        names = {name for name in self._authorized_names if name in self.catalog.by_name
                 and authorized(self.catalog.by_name[name])}
        if names != self._authorized_names:
            self._authorized_names = names
            self._revision += 1

    def permitted(self, name: str, *, authorized: Callable[[ToolCatalogEntry], bool]) -> bool:
        """Execution-boundary permission; an old visible snapshot cannot regrant."""
        with self._lock:
            self._tighten(authorized)
            entry = self.catalog.by_name.get(name)
            return entry is not None and self._eligible(entry, authorized, lambda _entry: True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return deepcopy({"version": 2, "revision": self._revision, "catalog_revision": self._catalog_revision,
                "ranker_version": self.ranker.version,
                "ranker_metadata": {name: self.catalog.by_name[name].ranking_hash for name in self._initial_catalog or {}
                                    if name in self.catalog.by_name},
                "initialized": self._initial_catalog is not None,
                "authorized_names": sorted(self._authorized_names),
                "initial_catalog": self._initial_catalog or {}, "discovered": self._discovered,
                "visible_seen": self._visible_seen, "search_ledger": self._ledger})

    def restore(self, state: Mapping[str, Any] | None, *, pending_tools: Sequence[str] = ()) -> None:
        if not state:
            return
        with self._lock:
            if state.get("version") != 2:
                raise ToolDiscoveryError("tool_catalog_changed", "Saved discovery state is incompatible")
            if state.get("ranker_version") != self.ranker.version:
                raise ToolDiscoveryError("tool_catalog_changed", "Saved tool ranking version is incompatible")
            initial = state.get("initial_catalog", {})
            if not isinstance(initial, Mapping):
                raise ToolDiscoveryError("tool_catalog_changed", "Saved tool identities are invalid")
            ranker_metadata = state.get("ranker_metadata", {})
            if (not isinstance(ranker_metadata, Mapping) or set(ranker_metadata) != set(initial)
                    or any(name in self.catalog.by_name and self.catalog.by_name[name].ranking_hash != digest
                           for name, digest in ranker_metadata.items())):
                raise ToolDiscoveryError("tool_catalog_changed", "Saved tool intent metadata changed")
            discovered = state.get("discovered", {})
            seen = state.get("visible_seen", {})
            for identities in (initial, discovered, seen):
                if not isinstance(identities, Mapping) or any(not isinstance(value, Mapping) or value.get("name") != name
                                                            for name, value in identities.items()):
                    raise ToolDiscoveryError("tool_catalog_changed", "Saved tool identities are invalid")
            protected = {**seen, **discovered}
            for name in pending_tools:
                if name in initial:
                    protected[name] = initial[name]
            if any(not self._compatible(identity) for identity in protected.values()):
                raise ToolDiscoveryError("tool_catalog_changed", "A previously exposed or pending tool changed; start a new Run")
            if self._initial_catalog is not None and int(state.get("revision", 0)) <= self._revision:
                return
            self._revision = int(state.get("revision", 0))
            self._catalog_revision = str(state.get("catalog_revision", self.catalog.revision))
            self._initial_catalog = deepcopy(dict(initial)) if state.get("initialized", True) else None
            self._authorized_names = set(state.get("authorized_names", initial)) & set(initial)
            self._discovered = deepcopy(dict(discovered))
            self._visible_seen = deepcopy(dict(seen))
            self._ledger = deepcopy(dict(state.get("search_ledger", {})))

    def plan(self, *, authorized: Callable[[ToolCatalogEntry], bool], available: Callable[[ToolCatalogEntry], bool],
             schema_budget: int, required_direct: Sequence[str] = (), optional_direct: Sequence[str] = (),
             discovery_enabled: bool = True, discovered_state: Mapping[str, Any] | None = None) -> ExposurePlan:
        with self._lock:
            if discovered_state:
                self.restore(discovered_state)
            if schema_budget < 1:
                raise ToolDiscoveryError("invalid_tool_schema_budget", "Tool schema budget must be positive")
            if self._initial_catalog is None:
                self._initial_catalog = {entry.name: entry.identity for entry in self.catalog.entries
                                         if entry.exposure != "hidden" and authorized(entry)}
                self._authorized_names = set(self._initial_catalog)
                self._revision += 1
            self._tighten(authorized)
            eligible = {entry.name: entry for entry in self.catalog.entries if self._eligible(entry, authorized, available)}
            required = set(required_direct) | {entry.name for entry in eligible.values()
                                              if entry.exposure == "direct" and entry.required}
            # Search disappears when no other discoverable capability remains.
            direct_names = required | set(optional_direct) | {entry.name for entry in eligible.values() if entry.exposure == "direct"}
            has_deferred = bool(set(eligible) - direct_names)
            if not has_deferred and discovery_enabled:
                # Even an all-direct catalog can need search when optional
                # definitions must be demoted for schema space.
                has_deferred = count_tokens_approximately([], tools=[entry.tool for entry in eligible.values()
                    if entry.name != "search_tools"]) > schema_budget
            if not discovery_enabled or not has_deferred:
                eligible.pop("search_tools", None)
                required.discard("search_tools")
            if required - set(eligible):
                raise ToolDiscoveryError("required_tool_unavailable", "A required tool is unavailable for this Run")
            accepted = []
            def add(entry: ToolCatalogEntry, *, must_fit: bool) -> bool:
                candidate = [*(item.tool for item in accepted), entry.tool]
                if count_tokens_approximately([], tools=candidate) > schema_budget:
                    if must_fit:
                        raise ToolDiscoveryError("tool_schema_budget_exhausted", "Required or previously discovered tool schemas exceed the budget")
                    return False
                accepted.append(entry)
                return True
            if not discovery_enabled:
                for entry in eligible.values():
                    add(entry, must_fit=True)
            else:
                mandatory = required | (set(self._discovered) & set(eligible))
                for name in sorted(mandatory):
                    add(eligible[name], must_fit=True)
                optional = [entry for entry in eligible.values() if entry.name not in mandatory
                            and (entry.exposure == "direct" or entry.name in optional_direct)]
                for entry in sorted(optional, key=lambda item: (-item.direct_priority, item.name)):
                    add(entry, must_fit=False)
            names = tuple(entry.name for entry in accepted)
            deferred = tuple(sorted((set(eligible) - set(names)) | (set(self._discovered) & set(eligible)))) if discovery_enabled else ()
            for entry in accepted:
                self._visible_seen[entry.name] = entry.identity
            self._revision += 1
            categories = sorted({eligible[name].category for name in deferred})
            summary = ""
            if categories:
                summary = ("可按需搜索的工具类别：" + "、".join(categories)
                           + "。未绑定的工具须先调用 search_tools，成功发现后在下一轮调用。"
                           if "search_tools" in names else "当前工具发现不可用，请使用本轮已绑定的能力完成任务。")
            return ExposurePlan(tuple(entry.tool for entry in accepted), names,
                tuple(entry.identity for entry in accepted), count_tokens_approximately([], tools=[entry.tool for entry in accepted]),
                schema_budget, summary, deferred, discovery_enabled)

    def search(self, query: str, *, call_id: str, plan: ExposurePlan,
               authorized: Callable[[ToolCatalogEntry], bool], available: Callable[[ToolCatalogEntry], bool],
               category: str | None = None, limit: int = 4) -> dict[str, Any]:
        with self._lock:
            if not isinstance(query, str) or not 1 <= len(query.strip()) <= 512 or type(limit) is not int or not 1 <= limit <= 8:
                raise ToolDiscoveryError("invalid_tool_search", "query must contain 1–512 characters and limit must be 1–8")
            if not call_id or not isinstance(call_id, str):
                raise ToolDiscoveryError("invalid_tool_search", "Search requires a trusted call identity")
            if call_id in self._ledger:
                raise ToolDiscoveryError("duplicate_tool_call", "This search call was already completed")
            if not plan.discovery_enabled or "search_tools" not in plan.names:
                raise ToolDiscoveryError("tool_not_exposed", "Tool discovery is not available in this model request")
            self._tighten(authorized)
            candidates = [self.catalog.by_name[name] for name in plan.deferred_names
                          if name != "search_tools" and name in self.catalog.by_name
                          and self._eligible(self.catalog.by_name[name], authorized, available)]
            categories = sorted({entry.category for entry in candidates})
            if category is not None:
                if not isinstance(category, str) or category not in categories:
                    raise ToolDiscoveryError("invalid_tool_category", "The requested tool category is unavailable")
                candidates = [entry for entry in candidates if entry.category == category]
            ranked = self.ranker.rank(query.strip(), tuple(candidates), limit=limit)
            eligible_by_name = {entry.name: entry for entry in candidates}
            if (len(ranked) > limit or any(not isinstance(entry, ToolCatalogEntry) for entry in ranked)
                    or len({entry.name for entry in ranked}) != len(ranked)
                    or any(eligible_by_name.get(entry.name) is not entry for entry in ranked)):
                raise ToolDiscoveryError("invalid_tool_ranker", "Tool ranker returned a tool outside the eligible catalog")
            current_names = set(plan.names) | set(self._discovered)
            current = [entry.tool for entry in self.catalog.entries if entry.name in current_names
                       and self._eligible(entry, authorized, available)]
            matches = []
            omitted = 0
            for entry in ranked:
                loaded = entry.name in current_names
                if not loaded and count_tokens_approximately([], tools=[*current, entry.tool]) > plan.schema_budget:
                    omitted += 1
                    continue
                if not loaded:
                    self._discovered[entry.name] = entry.identity
                    current_names.add(entry.name)
                    current.append(entry.tool)
                description = entry.when_to_use.strip() or entry.description.strip().split("\n\n", 1)[0]
                if len(description) > 320:
                    description = description[:319] + "…"
                matches.append({"name": entry.name, "version": entry.version,
                    "description": description, "loaded": True, "already_loaded": loaded})
            status = "loaded" if matches else "tool_schema_budget_exhausted" if ranked else "no_match"
            receipt: dict[str, Any] = {"catalog_revision": self._catalog_revision, "matches": matches,
                                     "status": status, "omitted_for_budget": omitted}
            if not ranked:
                receipt["available_categories"] = categories
            # Raw query text can contain business data and is not persisted here.
            self._ledger[call_id] = deepcopy(receipt)
            self._revision += 1
            return receipt


__all__ = ["DiscoveryCoordinator", "ExposurePlan", "ToolCatalog", "ToolCatalogEntry", "ToolDiscoveryError",
           "ToolRanker", "IntentClusterRanker",
           "merge_discovery_state", "tokenize_search_text"]
