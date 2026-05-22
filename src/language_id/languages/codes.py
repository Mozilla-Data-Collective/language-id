"""Thin wrappers around `langcodes`. Canonical form is BCP-47 (Common Voice standard)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from langcodes import Language, standardize_tag

_OVERRIDES_PATH = Path(__file__).parent / "_mapping_overrides.yaml"


@lru_cache(maxsize=1)
def _load_overrides() -> dict[str, str]:
    if not _OVERRIDES_PATH.exists():
        return {}
    data = yaml.safe_load(_OVERRIDES_PATH.read_text()) or {}
    mappings = data.get("mappings") or {}
    return {str(k): str(v) for k, v in mappings.items()}


def to_bcp47(code: str) -> str:
    """Return the canonical BCP-47 form of `code`.

    Applies the override table from `_mapping_overrides.yaml` first, then
    falls back to `langcodes.standardize_tag`.
    """
    code = code.strip()
    overrides = _load_overrides()
    if code in overrides:
        return overrides[code]
    return standardize_tag(code)


def language_name(code: str) -> str:
    """Return the English language name for a BCP-47 code."""
    return Language.get(to_bcp47(code)).display_name("en")
