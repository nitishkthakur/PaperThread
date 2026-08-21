"""Provider registry: config name -> adapter class.

Adding a provider means writing an adapter, registering it here, and adding a config
block. No calling code changes (REQUIREMENTS.md §11 D9).
"""

from __future__ import annotations

from ..config import Config
from .base import BasePaperProvider, Capability
from .http_cache import HTTPCache
from .papers.arxiv import ArxivProvider
from .papers.crossref import CrossrefProvider
from .papers.openalex import OpenAlexProvider
from .papers.semantic_scholar import SemanticScholarProvider

_PAPER_PROVIDERS: dict[str, type[BasePaperProvider]] = {
    "arxiv": ArxivProvider,
    "crossref": CrossrefProvider,
    "openalex": OpenAlexProvider,
    "semantic_scholar": SemanticScholarProvider,
}


class UnknownProviderError(RuntimeError):
    pass


def build_paper_providers(
    config: Config,
    capability: Capability | None = None,
    cache: HTTPCache | None = None,
) -> list[BasePaperProvider]:
    """Instantiate every enabled provider offering `capability`.

    Several providers answering the same capability is the normal case, not an edge case.
    """
    providers: list[BasePaperProvider] = []
    for entry in config.enabled_paper_providers(capability.value if capability else None):
        adapter_cls = _PAPER_PROVIDERS.get(entry.name)
        if adapter_cls is None:
            raise UnknownProviderError(
                f"config enables paper provider {entry.name!r} but no adapter is registered. "
                f"Known: {', '.join(sorted(_PAPER_PROVIDERS))}"
            )
        providers.append(adapter_cls(entry, cache=cache))
    return providers


def known_paper_providers() -> list[str]:
    return sorted(_PAPER_PROVIDERS)
