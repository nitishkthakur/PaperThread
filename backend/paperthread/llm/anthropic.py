"""Anthropic Messages API adapter.

Not the default (D10 makes that Ollama Cloud), but present because D9 requires the boundary
to be genuinely multi-provider rather than nominally so — an interface with one
implementation has not been shown to abstract anything.

Structured output uses a **forced tool call** rather than prompting for JSON. The schema
becomes the tool's `input_schema` and `tool_choice` names it, so the API constrains the
output shape instead of the prompt asking politely for it.

Note the asymmetry this file makes concrete: Anthropic has **no embeddings API**, so this
provider can never serve the embedding port. That is exactly why the two are separate ports
(PROVIDER_NOTES Part 2).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from .base import BaseLLMProvider, LLMError, LLMRequest

_API_VERSION = "2023-06-01"
_TOOL_NAME = "emit_result"


class AnthropicProvider(BaseLLMProvider):
    async def _complete(self, request: LLMRequest, model: str, user: str) -> str:
        key = self.require_key()
        url = self.config.base_url.rstrip("/") + "/messages"
        headers = {
            "x-api-key": key,
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "system": request.system,
            "messages": [{"role": "user", "content": user}],
            "tools": [
                {
                    "name": _TOOL_NAME,
                    "description": "Return the result. Call this exactly once.",
                    "input_schema": request.schema,
                }
            ],
            # Forcing the tool is what makes this structured output rather than a request
            # for structured output.
            "tool_choice": {"type": "tool", "name": _TOOL_NAME},
        }

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                response = await client.post(url, headers=headers, json=body)
        except httpx.HTTPError as exc:
            raise LLMError(self.name, f"request failed: {exc}") from exc

        if response.status_code == 401:
            raise LLMError(self.name, f"unauthorized — check {self.config.api_key_env}")
        if response.status_code == 429:
            raise LLMError(self.name, "rate limited (HTTP 429)")
        if response.status_code >= 400:
            raise LLMError(self.name, f"HTTP {response.status_code}: {response.text[:300]}")

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMError(self.name, f"malformed JSON: {exc}") from exc

        for block in payload.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == _TOOL_NAME:
                # Re-serialised so the shared validate-and-repair loop sees the same string
                # shape it gets from every other provider.
                return json.dumps(block.get("input", {}))

        raise LLMError(self.name, "model returned no tool_use block despite a forced tool")
