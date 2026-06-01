from typing import Literal

import regex

LengthBucket = Literal["short", "medium", "long"]

# Per-language scaling factors mapping character count -> approximate word count
# for languages without whitespace word boundaries.
_CHARS_PER_WORD_OVERRIDES: dict[str, float] = {
    "zh": 1.5,
    "zh-Hans": 1.5,
    "zh-Hant": 1.5,
    "ja": 1.5,
    "th": 5.0,
    "km": 5.0,
    "lo": 5.0,
    "my": 5.0,
    "bo": 5.0,
}

_WORD_PATTERN = regex.compile(r"\p{L}+")



def count_words(text: str, lang_code: str) -> int:
    """Return an approximate word count for `text`.

    - Default: unicode-aware word-token count via `regex` (`\\p{L}+`).
    - For non-spaced scripts (zh*, ja, th, km, lo, my, bo): apply an in-module
      `chars_per_word` factor. Falls back to the primary language subtag
      (e.g. "zh-CN" -> "zh").
    """
    factor = _CHARS_PER_WORD_OVERRIDES.get(lang_code)
    if factor is None:
        primary = lang_code.split("-")[0]
        factor = _CHARS_PER_WORD_OVERRIDES.get(primary)
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
