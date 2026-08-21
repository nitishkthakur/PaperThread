"""LLM registry and the client callers actually use.

`LLMClient` is the seam between the pipeline and any vendor. It owns three things the
pipeline must never do for itself:

* **model selection per task role** — topic decomposition is cheap and frequent while
  prerequisite judgment is expensive and quality-critical, so one global model prices one
  of them wrong (D10);
* **caching**, stamped with `{provider, model, prompt_version}` (D2/D10);
* **concurrency limiting**, so a path with 40 judgments does not open 40 sockets.

`available()` is the graceful-degradation check. Callers ask it and fall back to structural
output; they never catch a missing-key exception to discover the layer is off (D12).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..config import Config
from .anthropic import AnthropicProvider
from .base import BaseLLMProvider, LLMError, LLMRequest, LLMResult
from .cache import LLMCache
from .openai_compatible import OpenAICompatibleProvider

logger = logging.getLogger(__name__)

# Keyed by the `kind` field in config, not by vendor name: three vendors share one wire
# format, and a fourth is a config block rather than a code change (D9).
_KINDS: dict[str, type[BaseLLMProvider]] = {
    "openai_compatible": OpenAICompatibleProvider,
    "anthropic": AnthropicProvider,
}


class UnknownLLMKindError(RuntimeError):
    pass


def known_llm_kinds() -> list[str]:
    return sorted(_KINDS)


def build_llm_provider(config: Config) -> BaseLLMProvider:
    entry = config.llm.active
    adapter = _KINDS.get(entry.kind)
    if adapter is None:
        raise UnknownLLMKindError(
            f"llm provider {entry.name!r} declares kind {entry.kind!r}, which has no "
            f"adapter. Known kinds: {', '.join(known_llm_kinds())}"
        )
    return adapter(entry, max_retries=config.llm.max_retries)


class LLMClient:
    def __init__(self, config: Config, provider: BaseLLMProvider | None = None) -> None:
        self.config = config
        self.settings = config.llm
        self._provider = provider
        self._cache = LLMCache(self.settings.cache_dir, enabled=self.settings.cache_enabled)
        self._semaphore = asyncio.Semaphore(max(1, self.settings.max_concurrency))

    @property
    def provider(self) -> BaseLLMProvider:
        if self._provider is None:
            self._provider = build_llm_provider(self.config)
        return self._provider

    def available(self) -> tuple[bool, str | None]:
        """Can L4 run right now? Returns (available, reason-if-not).

        Checked up front so a path can be built *without* the LLM and say so, rather than
        failing halfway through and leaving the user with nothing.
        """
        if not self.config.retrieval.layers.llm:
            return False, "L4 is disabled in config ([retrieval.layers] llm = false)."
        try:
            entry = self.config.llm.active
        except Exception as exc:  # ConfigError
            return False, str(exc)
        if entry.needs_key and not entry.api_key:
            return False, (
                f"LLM provider {entry.name!r} needs {entry.api_key_env} in the "
                f"environment; it is not set."
            )
        if entry.kind not in _KINDS:
            return False, f"no adapter for LLM kind {entry.kind!r}."
        return True, None

    async def structured(
        self,
        *,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        prompt_version: str,
        max_tokens: int = 8192,
    ) -> LLMResult:
        """Run a structured call, walking the role's model chain until one succeeds.

        The chain is config (`[llm.roles]`): primary first, then fallbacks. A model is
        abandoned when it is unreachable, missing from the endpoint, rate limited, or
        cannot produce output matching the schema within its repair retries — all of which
        mean "this model cannot do the job right now", which is what a fallback is for.

        **Every model in the chain is checked against the cache before any call is made.**
        A path already built with the second-choice model must not be recomputed just
        because the first choice came back.
        """
        chain = self.settings.models_for(role)
        if not chain:
            raise LLMError(self.settings.provider, f"no model configured for role {role!r}")

        def cache_key(model: str) -> str:
            return self._cache.key(
                provider=self.settings.provider,
                model=model,
                prompt_version=prompt_version,
                role=role,
                system=system,
                user=user,
            )

        for model in chain:
            if cached := self._cache.get(cache_key(model)):
                return LLMResult(
                    data=cached["data"],
                    model=cached.get("model", model),
                    provider=cached.get("provider", self.settings.provider),
                    prompt_version=cached.get("prompt_version", prompt_version),
                    cached=True,
                    fell_back=model != chain[0],
                )

        request = LLMRequest(
            role=role,
            system=system,
            user=user,
            schema=schema,
            prompt_version=prompt_version,
            max_tokens=max_tokens,
        )

        failures: list[str] = []
        for index, model in enumerate(chain):
            try:
                async with self._semaphore:
                    data = await self.provider.structured(request, model)
            except LLMError as exc:
                failures.append(f"{model}: {exc}")
                logger.warning(
                    "role %s: %s failed (%s); %s",
                    role,
                    model,
                    exc,
                    f"falling back to {chain[index + 1]}" if index + 1 < len(chain) else "no fallback left",
                )
                continue

            self._cache.put(
                cache_key(model),
                {
                    "data": data,
                    "provider": self.settings.provider,
                    "model": model,
                    "prompt_version": prompt_version,
                    "role": role,
                },
            )
            return LLMResult(
                data=data,
                model=model,
                provider=self.settings.provider,
                prompt_version=prompt_version,
                fell_back=index > 0,
            )

        raise LLMError(
            self.settings.provider,
            f"every model for role {role!r} failed — " + "; ".join(failures),
        )
