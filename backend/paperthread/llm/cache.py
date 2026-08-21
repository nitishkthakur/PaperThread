"""Content-addressed cache for LLM artifacts.

D2 is explicit that stage-2 output is **persisted data, not transient model output**:
prerequisite edges and explanations are recomputed only when their inputs change, never
per request. Recomputing them would make a learning path reshuffle between page loads —
§6's incremental updates assume the path is stable — and would put an unbounded LLM bill
on the request path.

Every entry is stamped with `{provider, model, prompt_version}` and that stamp is part of
the key. Changing the default model therefore *misses* rather than silently serving
judgments a different model made (D10). The stamp is stored in the payload too, so a cached
explanation can always be attributed in the UI.

This is a file cache because the database does not exist yet. The key discipline is the
part that matters and it ports unchanged to a table; the storage medium does not.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class LLMCache:
    def __init__(self, directory: Path, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def key(
        self,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        role: str,
        system: str,
        user: str,
    ) -> str:
        """Hash everything that could change the answer.

        The prompt text is hashed as well as its version, so editing a prompt without
        remembering to bump `prompt_version` still invalidates correctly. Version bumps are
        for humans reading provenance; the hash is what protects the cache.
        """
        digest = hashlib.sha256()
        for part in (provider, model, prompt_version, role, system, user):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def _path(self, key: str) -> Path:
        # Two-level fan-out: a topic-heavy session can produce thousands of entries, and
        # some filesystems degrade badly on flat directories that size.
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        path = self._path(key)
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            # A corrupt entry is a cache miss, never an error. Losing a cached judgment
            # costs one API call; failing the request costs the user their path.
            logger.warning("discarding unreadable cache entry %s: %s", path, exc)
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a crash mid-write must not leave a truncated entry that
            # reads as valid JSON.
            temporary = path.with_suffix(".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            temporary.replace(path)
        except OSError as exc:
            logger.warning("could not write cache entry %s: %s", path, exc)
