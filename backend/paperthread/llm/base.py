"""The LLM port, and the structured-output contract every adapter must honour.

`LLMProvider` is a **separate port from embeddings**, and that separation is load-bearing
rather than tidy-minded: Ollama Cloud — our default — offers no embedding models at all,
and Anthropic has no embeddings API. A single "AI provider" abstraction would break on the
first two providers we tried, not eventually (D10, PROVIDER_NOTES Part 2).

One method matters: `structured`. **Raw model text is never trusted.** Every call declares
a JSON schema, the response is validated against it, and a failure is fed back to the
model as a repair instruction for a bounded number of retries. Per-provider strategy lives
behind this method — native JSON-schema modes, plain JSON modes, and forced tool calls all
produce the same validated dict to the caller (D10).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..config import LLMProviderConfig


class LLMError(RuntimeError):
    """An LLM call failed. Never fatal: L4 degrades to structural output (D12)."""

    def __init__(self, provider: str, message: str) -> None:
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class SchemaError(ValueError):
    """Model output did not match the declared schema."""


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """One structured-output call.

    `prompt_version` is not decoration. It is stored with every cached artifact so that
    editing a prompt invalidates exactly the judgments that prompt produced, instead of
    leaving a cache full of results no current code path would generate (D10).
    """

    role: str
    system: str
    user: str
    schema: dict[str, Any]
    prompt_version: str
    temperature: float = 0.0
    max_tokens: int = 8192


@dataclass(frozen=True, slots=True)
class LLMResult:
    data: dict[str, Any]
    model: str
    provider: str
    prompt_version: str
    # True when this came from the cache rather than the provider.
    cached: bool = False
    # True when the role's primary model failed and a fallback answered. Surfaced so a
    # path can say which model actually produced its judgments (D10).
    fell_back: bool = False

    def stamp(self) -> str:
        return f"{self.provider}/{self.model}/p{self.prompt_version}"


@runtime_checkable
class LLMPort(Protocol):
    name: str

    async def structured(self, request: LLMRequest, model: str) -> dict[str, Any]: ...


class BaseLLMProvider:
    """Shared plumbing: identity, credentials, and the validate-and-retry loop."""

    name: str = "base"

    def __init__(self, config: LLMProviderConfig, max_retries: int = 2) -> None:
        self.config = config
        self.name = config.name
        self.max_retries = max_retries

    def require_key(self) -> str:
        """The API key, or an error naming the variable config said to read it from.

        Only call this when `config.needs_key` is true — a provider with no `api_key_env`
        (the local Ollama daemon) authenticates by other means, and demanding a key would
        report a working provider as broken.
        """
        key = self.config.api_key
        if not key:
            raise LLMError(
                self.name,
                f"no API key: set {self.config.api_key_env} in the environment "
                f"(config names it, config never stores it)",
            )
        return key

    async def structured(self, request: LLMRequest, model: str) -> dict[str, Any]:
        """Call the model, validate, and repair on failure.

        The repair turn matters more than the retry count: handing the model its own
        malformed output plus the specific validation error recovers most failures on the
        first attempt, where a blind retry at temperature 0 would reproduce the same
        mistake exactly.
        """
        attempt = 0
        repair: str | None = None
        last_error: str = ""

        while attempt <= self.max_retries:
            user = request.user if repair is None else f"{request.user}\n\n{repair}"
            raw = await self._complete(request, model, user)
            try:
                payload = extract_json(raw)
                validate(payload, request.schema)
                return payload
            except (SchemaError, ValueError) as exc:
                last_error = str(exc)
                repair = (
                    "Your previous response was rejected.\n"
                    f"Error: {last_error}\n"
                    "Reply with ONLY a JSON object matching the schema. No prose, no "
                    "markdown fences, no explanation outside the JSON."
                )
                attempt += 1

        raise LLMError(
            self.name, f"invalid structured output after {self.max_retries + 1} attempts: {last_error}"
        )

    async def _complete(self, request: LLMRequest, model: str, user: str) -> str:
        raise NotImplementedError


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> dict[str, Any]:
    """Recover a JSON object from a model response.

    Models wrap JSON in markdown fences and prose despite being told not to, and some
    reasoning models emit a preamble before the payload. This is tolerant about the
    wrapper and strict about the content: whatever is extracted is still schema-validated
    by the caller, so leniency here never becomes leniency about correctness.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    candidates: list[str] = []
    if match := _FENCE_RE.search(text):
        candidates.append(match.group(1))
    candidates.append(text.strip())

    # Last resort: the outermost braces. Models that prepend commentary land here.
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")

    raise ValueError(f"response is not JSON: {text[:200]!r}")


def validate(payload: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate against the JSON Schema subset this codebase actually uses.

    Hand-rolled rather than pulling in `jsonschema`, because the supported subset is small
    and fixed — object/array/string/number/integer/boolean, `required`, `enum`, `items`,
    `properties`, and numeric bounds — and the error messages here are written to be fed
    back to a model as a repair instruction, which a generic validator's are not.
    """
    expected = schema.get("type")

    if expected == "object":
        if not isinstance(payload, dict):
            raise SchemaError(f"{path}: expected an object, got {_kind(payload)}")
        properties = schema.get("properties", {})
        for name in schema.get("required", []):
            if name not in payload:
                raise SchemaError(f"{path}: missing required field {name!r}")
        for name, value in payload.items():
            if name in properties:
                validate(value, properties[name], f"{path}.{name}")

    elif expected == "array":
        if not isinstance(payload, list):
            raise SchemaError(f"{path}: expected an array, got {_kind(payload)}")
        if (minimum := schema.get("minItems")) is not None and len(payload) < minimum:
            raise SchemaError(f"{path}: expected at least {minimum} items, got {len(payload)}")
        if item_schema := schema.get("items"):
            for i, item in enumerate(payload):
                validate(item, item_schema, f"{path}[{i}]")

    elif expected == "string":
        if not isinstance(payload, str):
            raise SchemaError(f"{path}: expected a string, got {_kind(payload)}")
        if (allowed := schema.get("enum")) and payload not in allowed:
            raise SchemaError(f"{path}: {payload!r} is not one of {allowed}")
        if (minimum := schema.get("minLength")) is not None and len(payload) < minimum:
            raise SchemaError(f"{path}: must be at least {minimum} characters")

    elif expected in {"number", "integer"}:
        # bool is a subclass of int in Python; a JSON boolean is not a JSON number.
        if isinstance(payload, bool) or not isinstance(payload, (int, float)):
            raise SchemaError(f"{path}: expected a {expected}, got {_kind(payload)}")
        if expected == "integer" and not float(payload).is_integer():
            raise SchemaError(f"{path}: expected an integer, got {payload}")
        if (minimum := schema.get("minimum")) is not None and payload < minimum:
            raise SchemaError(f"{path}: must be >= {minimum}")
        if (maximum := schema.get("maximum")) is not None and payload > maximum:
            raise SchemaError(f"{path}: must be <= {maximum}")

    elif expected == "boolean":
        if not isinstance(payload, bool):
            raise SchemaError(f"{path}: expected a boolean, got {_kind(payload)}")


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    return {
        bool: "boolean",
        int: "number",
        float: "number",
        str: "string",
        list: "array",
        dict: "object",
    }.get(type(value), type(value).__name__)
