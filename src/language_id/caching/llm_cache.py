"""Disk-backed LLM cache keyed by (model, version, prompt_template_hash, text_hash) (spec §5)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class LLMCache:
    """Thin wrapper around `diskcache.Cache` with structured key construction."""

    def __init__(self, cache_dir: Path) -> None:
        from diskcache import Cache

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Any = Cache(str(self.cache_dir))

    @staticmethod
    def make_key(model: str, version: str, prompt_hash: str, text_hash: str) -> str:
        return f"{model}|{version}|{prompt_hash}|{text_hash}"

    @staticmethod
    def hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def hash_prompt(*parts: str) -> str:
        return hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()[:16]

    def get(self, key: str) -> str | None:
        value = self._cache.get(key)
        return value if value is None else str(value)

    def set(self, key: str, value: str) -> None:
        self._cache.set(key, value)

    def clear(self, model: str | None = None) -> int:
        if model is None:
            n = len(self._cache)
            self._cache.clear()
            return n
        prefix = f"{model}|"
        removed = 0
        for k in list(self._cache):
            if isinstance(k, str) and k.startswith(prefix):
                del self._cache[k]
                removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        return {"size": len(self._cache)}
