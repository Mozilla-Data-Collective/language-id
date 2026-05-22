"""Word-count bucketing utilities for length-stratified sampling (spec §7.2)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import regex
import yaml

LengthBucket = Literal["short", "medium", "long"]

# configs/word_count_overrides.yaml lives at the repo root, alongside pyproject.toml.
# This module sits at src/language_id/data/length_buckets.py, so the repo root is parents[3].
_OVERRIDES_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "word_count_overrides.yaml"
)
_WORD_PATTERN = regex.compile(r"\p{L}+")


@lru_cache(maxsize=1)
def _load_chars_per_word() -> dict[str, float]:
    if not _OVERRIDES_PATH.exists():
        return {}
    data = yaml.safe_load(_OVERRIDES_PATH.read_text()) or {}
    raw = data.get("chars_per_word") or {}
    return {str(k): float(v) for k, v in raw.items()}


def count_words(text: str, lang_code: str) -> int:
    """Return an approximate word count for `text`.

    - Default: unicode-aware word-token count via `regex` (`\\p{L}+`).
    - For non-spaced scripts (zh*, ja, th, km, lo, my, bo): apply a per-language
      `chars_per_word` factor from `configs/word_count_overrides.yaml`. Falls
      back to the primary language subtag (e.g. "zh-CN" → "zh").
    """
    overrides = _load_chars_per_word()
    factor = overrides.get(lang_code)
    if factor is None:
        primary = lang_code.split("-")[0]
        factor = overrides.get(primary)
    if factor is not None:
        letter_count = sum(len(m) for m in _WORD_PATTERN.findall(text))
        return int(round(letter_count / factor))
    return len(_WORD_PATTERN.findall(text))


def assign_bucket(
    word_count: int,
    buckets: dict[str, tuple[int, int]],
) -> str | None:
    """Map a word count to a bucket name using inclusive-lower, exclusive-upper bounds.

    Returns None if `word_count` falls outside every bucket.
    """
    for name, (lo, hi) in buckets.items():
        if lo <= word_count < hi:
            return name
    return None
