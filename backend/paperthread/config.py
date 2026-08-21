"""Configuration loading.

The TOML file is the ONE place providers and models are selected (REQUIREMENTS.md §11 D9).
No provider choice is hardcoded and none is read from scattered environment lookups —
environment variables are used only for secrets, and only via the `*_env` indirection
named in config.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "paperthread.toml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or self-contradictory."""


@dataclass(frozen=True)
class PaperProviderConfig:
    name: str
    enabled: bool
    capabilities: frozenset[str]
    options: dict[str, object] = field(default_factory=dict)
    rate_limit_per_sec: float = 1.0
    timeout_seconds: float = 20.0
    api_key_env: str | None = None

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


@dataclass(frozen=True)
class LLMProviderConfig:
    name: str
    kind: str
    base_url: str
    api_key_env: str | None = None
    timeout_seconds: float = 120.0
    # How this endpoint constrains output. Not cosmetic: some endpoints ACCEPT a stricter
    # mode and silently ignore it, which burns a repair retry on every call rather than
    # failing loudly. See the comments in config/paperthread.toml.
    structured_output: str = "json_object"

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None

    @property
    def needs_key(self) -> bool:
        """A provider only needs a key if config named a variable to read one from.

        The local Ollama daemon needs none — it authenticates to Ollama Cloud with stored
        CLI credentials — and treating that as a missing key would report a working
        provider as unavailable.
        """
        return self.api_key_env is not None


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    # role -> model name, or role -> ordered [primary, *fallbacks].
    roles: dict[str, str | list[str]]
    providers: dict[str, LLMProviderConfig]
    # D2: stage-2 output is persisted data, not transient model output. Until the database
    # lands this is a content-addressed file cache; the stamp it stores is the same one a
    # table would.
    cache_enabled: bool = True
    cache_dir: Path = Path(".cache/llm")
    # Raw model text is never trusted: output is validated against a schema and retried
    # with the validation error fed back before the call is given up on.
    max_retries: int = 2
    max_concurrency: int = 4

    def models_for(self, role: str) -> list[str]:
        """The ordered model chain for a role: primary first, then fallbacks.

        Model is selected per task role, not globally (D10) — topic decomposition is cheap
        and frequent while prerequisite judgment is expensive and quality-critical, so one
        global model prices one of them wrong.

        Config may give a bare string (primary only) or a list (primary + fallbacks). The
        fallbacks matter because a hosted model can be retired, renamed, or briefly
        unreachable, and a learning path that fails entirely because one model moved is a
        worse outcome than one built by the second-choice model and labelled as such.
        """
        value = self.roles.get(role) or self.roles.get("default") or []
        if isinstance(value, str):
            return [value] if value else []
        return [model for model in value if model]

    def model_for(self, role: str) -> str:
        """The primary model for a role. Prefer `models_for` unless fallback is irrelevant."""
        chain = self.models_for(role)
        return chain[0] if chain else ""

    @property
    def active(self) -> LLMProviderConfig:
        try:
            return self.providers[self.provider]
        except KeyError:
            raise ConfigError(
                f"llm.provider is {self.provider!r} but [llm.providers.{self.provider}] is not defined"
            ) from None


@dataclass(frozen=True)
class EmbeddingsConfig:
    """Separate from LLMConfig, deliberately.

    Ollama Cloud has no embedding models and Anthropic has no embeddings API, so the
    default LLM provider cannot serve embeddings (PROVIDER_NOTES L1).
    """

    enabled: bool
    provider: str
    model: str
    dimensions: int
    model_revision: str

    @property
    def vector_stamp(self) -> str:
        """Recorded with every stored vector; a change means the index must be rebuilt."""
        return f"{self.provider}/{self.model}@{self.model_revision}/{self.dimensions}"


@dataclass(frozen=True)
class LayerConfig:
    """D12: each layer is independently disableable and must degrade, not break."""

    lexical: bool = True
    local_nlp: bool = False
    embeddings: bool = False
    reranking: bool = False
    llm: bool = False


@dataclass(frozen=True)
class ExpansionConfig:
    """Stage 2 budget. Every knob here costs provider requests, so it is config, not code."""

    enabled: bool = True
    seed_papers: int = 25
    references_per_paper: int = 60
    min_co_citations: int = 3
    max_ancestors: int = 20
    expand_ancestors: bool = True
    ancestor_seeds: int = 12
    forward_enabled: bool = True
    forward_seeds: int = 6
    citations_per_paper: int = 25


@dataclass(frozen=True)
class GraphConfig:
    """Stage 3 parameters, all evidence-backed — see docs/RETRIEVAL_NOTES.md."""

    # Chen et al. (2007) use d=0.5 for citation networks, not the web-standard 0.85:
    # ~42-51% of a bibliography's references cite each other, so reference-following paths
    # are short. At d=0.9 PageRank degenerates toward raw citation count.
    pagerank_damping: float = 0.5
    pagerank_iterations: int = 100
    pagerank_tolerance: float = 1e-9
    # Age-rescaling cohort half-width, in years. PageRank is compared only against papers
    # of a similar age, because raw PageRank "completely fails to identify recent milestone
    # papers" (Mariani, Medo & Zhang 2016).
    age_cohort_years: int = 3
    min_cohort_size: int = 5
    # Louvain resolution. 1.0 is standard modularity.
    community_resolution: float = 1.0
    max_path_papers: int = 24


@dataclass(frozen=True)
class RetrievalConfig:
    candidates_per_provider: int
    max_candidates: int
    layers: LayerConfig
    rrf_k: int
    # Which path builder runs. "structural" is the no-LLM baseline; everything else is an
    # LLM-planned strategy from retrieval/curriculum.py.
    path_strategy: str = "syllabus"
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)


@dataclass(frozen=True)
class ProviderCacheConfig:
    enabled: bool = True
    cache_dir: Path = Path(".cache/http")


@dataclass(frozen=True)
class Config:
    default_user_id: str
    paper_providers: tuple[PaperProviderConfig, ...]
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    retrieval: RetrievalConfig
    source_path: Path
    provider_cache: ProviderCacheConfig = field(default_factory=ProviderCacheConfig)

    def enabled_paper_providers(self, capability: str | None = None) -> list[PaperProviderConfig]:
        """Providers are selected by CAPABILITY, not by vendor (D11).

        Multiple providers answering the same capability is the normal case.
        """
        providers = [p for p in self.paper_providers if p.enabled]
        if capability is not None:
            providers = [p for p in providers if capability in p.capabilities]
        return providers


_RESERVED_PROVIDER_KEYS = {
    "name",
    "enabled",
    "capabilities",
    "rate_limit_per_sec",
    "timeout_seconds",
    "api_key_env",
}


_T = TypeVar("_T")

_COERCIONS: dict[str, type] = {"bool": bool, "int": int, "float": float, "str": str}


def _dataclass_from(cls: type[_T], raw: dict[str, Any], defaults: _T) -> _T:
    """Build a flat config dataclass from a TOML table, coercing by declared field type.

    An UNKNOWN key is an error, not a shrug. These tables are tuning knobs: a typo that
    silently leaves the default in place would look exactly like "the setting had no
    effect", which is the most expensive kind of configuration bug to chase.
    """
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    if unknown := sorted(set(raw) - set(known)):
        raise ConfigError(
            f"unknown key(s) in config: {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(known))}"
        )

    values: dict[str, Any] = {}
    for name, spec in known.items():
        if name not in raw:
            continue
        coerce = _COERCIONS.get(str(spec.type), lambda v: v)
        try:
            values[name] = coerce(raw[name])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"config key {name!r} is not a valid {spec.type}: {exc}") from exc
    return type(defaults)(**{**{f.name: getattr(defaults, f.name) for f in fields(cls)}, **values})  # type: ignore[arg-type]


def load_config(path: Path | str | None = None) -> Config:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")

    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    paper_providers = tuple(
        PaperProviderConfig(
            name=entry["name"],
            enabled=bool(entry.get("enabled", True)),
            capabilities=frozenset(entry.get("capabilities", [])),
            options={k: v for k, v in entry.items() if k not in _RESERVED_PROVIDER_KEYS},
            rate_limit_per_sec=float(entry.get("rate_limit_per_sec", 1.0)),
            timeout_seconds=float(entry.get("timeout_seconds", 20.0)),
            api_key_env=entry.get("api_key_env"),
        )
        for entry in raw.get("paper_providers", [])
    )

    llm_raw = raw.get("llm", {})
    cache_raw = llm_raw.get("cache", {})
    cache_dir = Path(cache_raw.get("dir", ".cache/llm"))
    if not cache_dir.is_absolute():
        cache_dir = config_path.resolve().parents[1] / cache_dir
    llm = LLMConfig(
        provider=llm_raw.get("provider", ""),
        roles=dict(llm_raw.get("roles", {})),
        cache_enabled=bool(cache_raw.get("enabled", True)),
        cache_dir=cache_dir,
        max_retries=int(llm_raw.get("max_retries", 2)),
        max_concurrency=int(llm_raw.get("max_concurrency", 4)),
        providers={
            name: LLMProviderConfig(
                name=name,
                kind=entry.get("kind", "openai_compatible"),
                base_url=entry.get("base_url", ""),
                api_key_env=entry.get("api_key_env"),
                timeout_seconds=float(entry.get("timeout_seconds", 120.0)),
                structured_output=entry.get("structured_output", "json_object"),
            )
            for name, entry in llm_raw.get("providers", {}).items()
        },
    )

    emb_raw = raw.get("embeddings", {})
    embeddings = EmbeddingsConfig(
        enabled=bool(emb_raw.get("enabled", False)),
        provider=emb_raw.get("provider", ""),
        model=emb_raw.get("model", ""),
        dimensions=int(emb_raw.get("dimensions", 0)),
        model_revision=emb_raw.get("model_revision", "main"),
    )

    ret_raw = raw.get("retrieval", {})
    layers_raw = ret_raw.get("layers", {})
    exp_raw = ret_raw.get("expansion", {})
    graph_raw = ret_raw.get("graph", {})
    defaults_expansion = ExpansionConfig()
    defaults_graph = GraphConfig()
    retrieval = RetrievalConfig(
        candidates_per_provider=int(ret_raw.get("candidates_per_provider", 25)),
        max_candidates=int(ret_raw.get("max_candidates", 300)),
        layers=LayerConfig(
            lexical=bool(layers_raw.get("lexical", True)),
            local_nlp=bool(layers_raw.get("local_nlp", False)),
            embeddings=bool(layers_raw.get("embeddings", False)),
            reranking=bool(layers_raw.get("reranking", False)),
            llm=bool(layers_raw.get("llm", False)),
        ),
        rrf_k=int(ret_raw.get("fusion", {}).get("rrf_k", 60)),
        path_strategy=str(ret_raw.get("path_strategy", "syllabus")),
        expansion=_dataclass_from(ExpansionConfig, exp_raw, defaults_expansion),
        graph=_dataclass_from(GraphConfig, graph_raw, defaults_graph),
    )

    pcache_raw = raw.get("providers", {}).get("cache", {})
    pcache_dir = Path(pcache_raw.get("dir", ".cache/http"))
    if not pcache_dir.is_absolute():
        pcache_dir = config_path.resolve().parents[1] / pcache_dir

    return Config(
        provider_cache=ProviderCacheConfig(
            enabled=bool(pcache_raw.get("enabled", True)), cache_dir=pcache_dir
        ),
        default_user_id=raw.get("app", {}).get("default_user_id", "local"),
        paper_providers=paper_providers,
        llm=llm,
        embeddings=embeddings,
        retrieval=retrieval,
        source_path=config_path,
    )
