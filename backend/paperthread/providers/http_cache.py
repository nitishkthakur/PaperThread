"""Disk cache for provider HTTP responses.

D11 requires that no provider call sits on a user request path un-cached, and the rate
limits make the reason concrete: arXiv asks for one request per three seconds, and a single
learning path makes tens of expansion requests. Without this, rebuilding a path for a topic
someone already asked about costs the same as building it the first time.

It is also what makes the ranking work tunable at all. Comparing five strategies across ten
topics is hundreds of provider requests per iteration; against a warm cache the same sweep
is bounded by the LLM calls instead of by OpenAlex's politeness policy.

Cached on the **request**, not the parsed result, so a parser change does not require
re-fetching. Entries never expire on their own: paper metadata is close to immutable, and a
stale citation count is a far smaller problem than an unusable iteration loop. Delete the
directory to refresh.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class HTTPCache:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

    def key(self, provider: str, url: str, params: dict[str, Any] | None = None) -> str:
        digest = hashlib.sha256()
        digest.update(provider.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(url.encode("utf-8"))
        if params:
            # Sorted so that dict ordering never produces two keys for one request.
            digest.update(
                json.dumps({k: str(v) for k, v in sorted(params.items())}).encode("utf-8")
            )
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        if not self.enabled:
            return None
        try:
            with self._path(key).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.hits += 1
            return payload
        except FileNotFoundError:
            self.misses += 1
            return None
        except (OSError, ValueError) as exc:
            # A corrupt entry is a miss, never an error: refetching costs one request.
            logger.warning("discarding unreadable cache entry for %s: %s", key, exc)
            self.misses += 1
            return None

    def put(self, key: str, payload: Any) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            temporary.replace(path)
        except OSError as exc:
            logger.warning("could not write cache entry %s: %s", path, exc)

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        return f"{self.hits} hits / {total} requests ({rate:.0f}%)"
