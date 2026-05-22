"""Shared LLM logic: retry, cache, parse (spec §3 Experiment 1)."""

from __future__ import annotations

import time

from language_id.caching.llm_cache import LLMCache
from language_id.models.base import LIDPrediction
from language_id.parsing.llm_output import parse_llm_output


class BaseLLMModel:
    """Base class for LLM-backed LID models.

    Subclasses implement `_raw_call(text)` returning the model's free-text response.
    The base class handles caching, retry-with-backoff, and parsing to BCP-47.
    """

    name: str = "llm"
    version: str = "TBD"

    def __init__(
        self,
        name: str,
        version: str,
        prompt_hash: str,
        cache: LLMCache | None = None,
        max_retries: int = 3,
        retry_backoff_s: float = 1.0,
    ) -> None:
        self.name = name
        self.version = version
        self.prompt_hash = prompt_hash
        self.cache = cache
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def predict(self, text: str) -> LIDPrediction:
        raw = self._call_with_cache(text)
        parsed = parse_llm_output(raw)
        return LIDPrediction(
            lang_code=parsed.lang_code if parsed.lang_code is not None else "und",
            confidence=None,
            raw_output=raw,
        )

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in texts]

    def _call_with_cache(self, text: str) -> str:
        if self.cache is None:
            return self._call_with_retry(text)
        key = LLMCache.make_key(
            self.name, self.version, self.prompt_hash, LLMCache.hash_text(text)
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw = self._call_with_retry(text)
        self.cache.set(key, raw)
        return raw

    def _call_with_retry(self, text: str) -> str:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._raw_call(text)
            except Exception as e:
                last_exc = e
                if attempt + 1 < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2**attempt))
        assert last_exc is not None
        raise last_exc

    def _raw_call(self, text: str) -> str:
        """Single LLM call. Subclasses must implement."""
        raise NotImplementedError
