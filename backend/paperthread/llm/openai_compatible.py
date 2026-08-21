"""OpenAI-compatible chat-completions adapter.

Covers the local Ollama daemon (which reaches Ollama Cloud models when signed in), hosted
Ollama Cloud, and OpenRouter — same wire format, three base URLs, one adapter. Adding
another OpenAI-compatible vendor is a config block, not code (D9).

Two behaviours here exist because of measured endpoint reality, not preference:

**Structured output is configured, not negotiated.** Verified 2026-08-16 against Ollama
0.32.12 with cloud models: `response_format: {"type": "json_schema"}` is accepted and
**silently ignored** — the model returns prose and the endpoint returns 200. Ollama's
native `format: <schema>` behaves the same way. A negotiate-on-error strategy therefore
never triggers, and every call pays a wasted repair retry. The mode comes from
`structured_output` in config, and `json_object` plus an explicit prompt instruction is
what actually works there.

**Reasoning models leave `content` empty.** Ollama's cloud models return their chain of
thought in `reasoning` (or `thinking`) and, when the token budget runs out mid-thought, an
empty `content` with `finish_reason: length`. That is a truncation bug that looks exactly
like a refusal, so it is detected and reported as truncation.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMError, LLMRequest

logger = logging.getLogger(__name__)


class OpenAICompatibleProvider(BaseLLMProvider):
    async def _complete(self, request: LLMRequest, model: str, user: str) -> str:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.config.needs_key:
            headers["Authorization"] = f"Bearer {self.require_key()}"

        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": user},
            ],
            # Judgments are cached and persisted (D2). Sampling would make the stored path
            # depend on when it happened to be computed.
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        mode = self.config.structured_output
        if mode == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": f"paperthread_{request.role}",
                    "strict": True,
                    "schema": request.schema,
                },
            }
        elif mode == "json_object":
            body["response_format"] = {"type": "json_object"}

        return await self._post(url, headers, body, model)

    async def _post(
        self, url: str, headers: dict[str, str], body: dict[str, Any], model: str
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(self.name, f"request failed for {model}: {exc}") from exc

        if response.status_code == 401:
            raise LLMError(
                self.name,
                f"unauthorized — check {self.config.api_key_env}. For the local daemon, "
                f"run `ollama signin`.",
            )
        if response.status_code == 404:
            # Distinct from a transport failure: this model is not installed here, which is
            # exactly the case the role's fallback chain exists for.
            raise LLMError(self.name, f"model {model!r} not found on this endpoint")
        if response.status_code == 429:
            raise LLMError(self.name, f"rate limited (HTTP 429) for {model}")
        if response.status_code >= 400:
            raise LLMError(
                self.name, f"HTTP {response.status_code} for {model}: {response.text[:300]}"
            )

        try:
            payload = response.json()
            choices = payload["choices"]
            if not choices:
                raise LLMError(self.name, f"{model} returned no choices")
            choice = choices[0]
            message = choice["message"]
        except (ValueError, KeyError, TypeError) as exc:
            raise LLMError(self.name, f"unexpected response shape from {model}: {exc}") from exc

        content = (message.get("content") or "").strip()
        if content:
            return content

        # Empty content on a reasoning model almost always means the token budget was spent
        # thinking. Saying so beats "invalid JSON", which sends the reader after the prompt.
        if choice.get("finish_reason") == "length":
            raise LLMError(
                self.name,
                f"{model} hit the token limit before emitting an answer — it is a "
                f"reasoning model and spent the budget on reasoning. Raise max_tokens.",
            )
        for key in ("reasoning_content", "reasoning", "thinking"):
            if fallback := (message.get(key) or "").strip():
                logger.info("%s: %s returned only %s; parsing that", self.name, model, key)
                return fallback
        raise LLMError(self.name, f"{model} returned an empty message")
