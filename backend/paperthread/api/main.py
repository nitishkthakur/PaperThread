"""FastAPI application.

The frontend talks to the backend over HTTP only — no local-only shortcuts a hosted
deployment couldn't support (D3). Every user-owned concept is keyed by `user_id` even
though only one user exists today.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..config import Config, ConfigError, load_config
from ..domain.models import RankedPaper
from ..domain.path import LearningPath, PathStep
from ..llm.registry import LLMClient, known_llm_kinds
from ..providers.registry import known_paper_providers
from ..retrieval.curriculum import STRATEGIES, BuildContext, build_strategy
from ..retrieval.path import LearningPathService
from ..retrieval.search import SearchResult, TopicSearchService

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

api = APIRouter(prefix="/api")


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()


class PaperOut(BaseModel):
    id: str
    title: str
    abstract: str | None = None
    year: int | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    citation_count: int | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    external_ids: list[str] = Field(default_factory=list)
    # Explicit so the UI can show what we actually hold, per D8's depth model.
    depth: str
    has_abstract: bool
    # Why this paper is here. Placeholder until Stage 4 (LLM judgment) exists — the
    # product requires a real explanation per §5, and this is NOT it yet.
    found_by: list[str] = Field(default_factory=list)
    score: float


class ProviderOut(BaseModel):
    provider: str
    ok: bool
    count: int = 0
    error: str | None = None


class SearchOut(BaseModel):
    topic: str
    count: int
    papers: list[PaperOut]
    providers: list[ProviderOut]
    layers_used: list[str]
    degraded: bool
    notes: list[str]


def _to_paper_out(ranked: RankedPaper) -> PaperOut:
    paper = ranked.paper
    return PaperOut(
        id=paper.canonical_id,
        title=paper.title,
        abstract=paper.abstract,
        year=paper.year,
        authors=paper.authors,
        venue=paper.venue,
        citation_count=paper.citation_count,
        pdf_url=paper.pdf_url,
        landing_url=paper.landing_url,
        external_ids=sorted(str(e) for e in paper.external_ids),
        depth=paper.depth.name.lower(),
        has_abstract=paper.has_abstract,
        found_by=ranked.found_by,
        score=round(ranked.score, 6),
    )


def _to_search_out(result: SearchResult) -> SearchOut:
    return SearchOut(
        topic=result.topic,
        count=len(result.papers),
        papers=[_to_paper_out(r) for r in result.papers],
        providers=[
            ProviderOut(provider=o.provider, ok=o.ok, count=o.count, error=o.error)
            for o in result.providers
        ],
        layers_used=result.layers_used,
        degraded=result.degraded,
        notes=result.notes,
    )


class ExplanationOut(BaseModel):
    """§5's four questions. `source` says whether they were reasoned or measured — the UI
    must never present a structural explanation as though a model wrote it."""

    why_it_matters: str
    what_it_assumes: str
    what_it_teaches: str
    why_for_you: str
    source: str
    asserted_by: str


class SignalsOut(BaseModel):
    co_citations: int
    pagerank: float
    age_rescaled_pagerank: float
    in_degree: int
    out_degree: int
    discovered_by_expansion: bool


class EdgeOut(BaseModel):
    prerequisite_id: str
    dependent_id: str
    source: str
    confidence: float | None = None
    reason: str | None = None
    asserted_by: str


class SubtopicOut(BaseModel):
    id: str
    label: str
    summary: str | None = None
    order: int
    named_by_llm: bool


class StepOut(BaseModel):
    order: int
    level: int
    paper: PaperOut
    role: str
    signals: SignalsOut
    explanation: ExplanationOut
    subtopic_id: str | None = None
    prerequisite_ids: list[str] = Field(default_factory=list)
    already_read: bool = False


class PathOut(BaseModel):
    topic: str
    count: int
    levels: int
    steps: list[StepOut]
    subtopics: list[SubtopicOut]
    edges: list[EdgeOut]
    layers_used: list[str]
    stages_run: list[str]
    degraded: bool
    # A weak path must not render like a strong one; see domain/path.py.
    confidence: float
    confidence_reasons: list[str]
    notes: list[str]


def _to_step_out(step: PathStep) -> StepOut:
    return StepOut(
        order=step.order,
        level=step.level,
        paper=_to_paper_out(
            RankedPaper(paper=step.paper, score=step.signals.lexical_score, found_by=[], ranks={})
        ),
        role=step.role.value,
        signals=SignalsOut(
            co_citations=step.signals.co_citations,
            pagerank=round(step.signals.pagerank, 8),
            age_rescaled_pagerank=round(step.signals.age_rescaled_pagerank, 4),
            in_degree=step.signals.in_degree,
            out_degree=step.signals.out_degree,
            discovered_by_expansion=step.signals.discovered_by_expansion,
        ),
        explanation=ExplanationOut(
            why_it_matters=step.explanation.why_it_matters,
            what_it_assumes=step.explanation.what_it_assumes,
            what_it_teaches=step.explanation.what_it_teaches,
            why_for_you=step.explanation.why_for_you,
            source=step.explanation.source.value,
            asserted_by=step.explanation.provenance.stamp(),
        ),
        subtopic_id=step.subtopic_id,
        prerequisite_ids=list(step.prerequisite_ids),
        already_read=step.already_read,
    )


def _to_path_out(path: LearningPath) -> PathOut:
    return PathOut(
        topic=path.topic,
        count=len(path.steps),
        levels=path.levels,
        steps=[_to_step_out(step) for step in path.steps],
        subtopics=[
            SubtopicOut(
                id=s.id,
                label=s.label,
                summary=s.summary,
                order=s.order,
                named_by_llm=s.named_by_llm,
            )
            for s in path.subtopics
        ],
        edges=[
            EdgeOut(
                prerequisite_id=e.prerequisite_id,
                dependent_id=e.dependent_id,
                source=e.source.value,
                confidence=e.confidence,
                reason=e.reason,
                asserted_by=e.provenance.stamp(),
            )
            for e in path.edges
        ],
        layers_used=path.layers_used,
        stages_run=path.stages_run,
        degraded=path.degraded,
        confidence=round(path.confidence, 2),
        confidence_reasons=path.confidence_reasons,
        notes=path.notes,
    )


@api.get("/health")
async def health(config: Config = Depends(get_config)) -> dict:
    llm_available, llm_reason = LLMClient(config).available()
    return {
        "status": "ok",
        "config": str(config.source_path),
        "enabled_paper_providers": [p.name for p in config.enabled_paper_providers()],
        "known_paper_providers": known_paper_providers(),
        "llm_provider": config.llm.provider,
        "known_llm_kinds": known_llm_kinds(),
        "path_strategy": config.retrieval.path_strategy,
        "known_strategies": [*sorted(STRATEGIES), "structural"],
        "llm_available": llm_available,
        "llm_unavailable_reason": llm_reason,
        "layers": config.retrieval.layers.__dict__,
    }


@api.get("/search", response_model=SearchOut)
async def search(
    topic: str = Query(..., min_length=1, description="Topic to build a reading path for"),
    limit: int = Query(30, ge=1, le=200),
    config: Config = Depends(get_config),
) -> SearchOut:
    service = TopicSearchService(config)
    try:
        result = await service.search(topic, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_search_out(result)


@api.get("/path", response_model=PathOut)
async def learning_path(
    topic: str = Query(..., min_length=1, description="Topic to build a reading path for"),
    limit: int = Query(24, ge=3, le=60, description="Maximum papers in the path"),
    strategy: str | None = Query(
        None,
        description="Override the configured path builder. One of: "
        + ", ".join([*sorted(STRATEGIES), "structural"]),
    ),
    config: Config = Depends(get_config),
) -> PathOut:
    """The product's actual endpoint: an ordered path, not a ranked list.

    Slow by nature — stage 2 makes tens of provider requests under per-provider rate
    limits, and stage 4 adds LLM calls when enabled. Judgments are cached (D2), so the
    second request for a topic is dominated by the provider round-trips alone.
    """
    requested = strategy or config.retrieval.path_strategy
    client = LLMClient(config)
    available, reason = client.available()

    # An LLM strategy without an LLM is not a strategy. Fall back to the structural
    # pipeline and say which one actually ran, rather than returning an empty path.
    if requested != "structural" and not available:
        result = await LearningPathService(config).build(topic, limit=limit)
        result.notes.insert(
            0,
            f"Requested the {requested!r} strategy, but L4 is unavailable ({reason}), so "
            f"this is the structural pipeline instead.",
        )
        return _to_path_out(result)

    try:
        if requested == "structural":
            result = await LearningPathService(config).build(topic, limit=limit)
        else:
            context = BuildContext.create(config, client)
            result = await build_strategy(requested, context).build(topic, limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _to_path_out(result)


def create_app() -> FastAPI:
    app = FastAPI(
        title="PaperThread API",
        version="0.1.0",
        description="What research paper should I read next, and why?",
    )
    # Local-first dev convenience. Tighten before any real deployment (D3).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.include_router(api)

    @app.on_event("startup")
    async def _validate_config() -> None:
        try:
            config = get_config()
        except ConfigError as exc:
            raise RuntimeError(f"invalid configuration: {exc}") from exc
        logger = logging.getLogger(__name__)
        logger.info("config: %s", config.source_path)
        logger.info(
            "paper providers enabled: %s",
            ", ".join(p.name for p in config.enabled_paper_providers()) or "(none)",
        )

    return app


app = create_app()
