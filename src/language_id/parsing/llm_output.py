"""Parse free-text LLM output into BCP-47 codes and track unparseable rate (spec §8)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import langcodes

from language_id.languages.codes import to_bcp47

# Common prefixes models slip in despite "respond with only the tag" instructions.
_PREFIX_PATTERNS = [
    r"^the\s+language\s+is\s*:?\s*",
    r"^language\s+code\s*:\s*",
    r"^language\s*:\s*",
    r"^bcp[- ]?47(?:\s+language)?\s+tag\s*:?\s*",
    r"^tag\s*:\s*",
]
_PREFIX_RE = re.compile("|".join(_PREFIX_PATTERNS), re.IGNORECASE)
_TRAILING_PUNCT = ".,;:!?\"')(][{}"

# Hand-written alias table. Maps lowercased free-text → BCP-47 (or None if it
# indicates "no language detected").
_ALIAS_TABLE: dict[str, str | None] = {
    "n/a": None,
    "none": None,
    "unknown": None,
    "undetermined": None,
    "mixed": None,
    "multiple": None,
}


@dataclass
class ParsedLLMOutput:
    lang_code: str | None       # None if unparseable
    raw: str
    method: str                 # "langcodes_find" | "standardize_tag" | "alias_table" | "unparseable"


def _strip(raw: str) -> str:
    text = raw.strip().strip("`").strip("\"'").strip()
    text = _PREFIX_RE.sub("", text).strip()
    return text.rstrip(_TRAILING_PUNCT).strip()


def parse_llm_output(raw: str) -> ParsedLLMOutput:
    """Parse free-text LLM output to BCP-47 (spec §8)."""
    text = _strip(raw)
    if not text:
        return ParsedLLMOutput(lang_code=None, raw=raw, method="unparseable")

    # 1. Treat as a tag (covers clean responses like "en", "zh-Hans") and apply
    #    overrides + langcodes.standardize_tag via to_bcp47.
    candidate = to_bcp47(text)
    if candidate and candidate not in {"und", text.lower()} and langcodes.tag_is_valid(candidate):
        return ParsedLLMOutput(lang_code=candidate, raw=raw, method="standardize_tag")

    # 2. Look up by name (handles "Spanish", "español", "Castilian").
    try:
        found = langcodes.find(text)
        return ParsedLLMOutput(lang_code=found.to_tag(), raw=raw, method="langcodes_find")
    except LookupError:
        pass

    # 3. Final pass: tag_is_valid on the lowercased token (catches "EN", "ZH").
    if langcodes.tag_is_valid(text):
        return ParsedLLMOutput(lang_code=to_bcp47(text), raw=raw, method="standardize_tag")

    # 4. Alias table.
    if text.lower() in _ALIAS_TABLE:
        return ParsedLLMOutput(
            lang_code=_ALIAS_TABLE[text.lower()], raw=raw, method="alias_table"
        )

    return ParsedLLMOutput(lang_code=None, raw=raw, method="unparseable")
