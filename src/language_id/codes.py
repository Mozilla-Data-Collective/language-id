"""
CommonLID labels languages with ISO-639-3 and CommonVoice LID label languages
with a mix of ISO-639-3, ISO-639-1 and BCP-47-style region/variant tags.
Everything normalizes to ISO-639-3 for comparison via `to_iso3`.

`lang_codes_mapping.csv` is the table, keyed by ISO-639-3 (column `iso639_3`).
Each row carries the `name`, the `iso639_1` 2-letter code (empty if none), and
the canonical `bcp47` tag. It covers the CommonLID + CommonVoice language sets.
"""


import csv
from functools import lru_cache
from pathlib import Path

import langcodes

LANG_TABLE_PATH = Path(__file__).parent / "lang_codes_mapping.csv"

UND = "und"  # undetermined / unparseable


@lru_cache(maxsize=1)
def table() -> dict[str, dict[str, str | None]]:
    """ISO-639-3 -> {name, iso639_1, bcp47}. Empty cells become None."""
    out: dict[str, dict[str, str | None]] = {}
    with LANG_TABLE_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            out[row["iso639_3"]] = {
                "name": row["name"] or None,
                "iso639_1": row["iso639_1"] or None,
                "bcp47": row["bcp47"] or None,
            }
    return out


@lru_cache(maxsize=1)
def _reverse_index() -> dict[str, str]:
    """Lowercased iso639_1 / bcp47 / name -> ISO-639-3."""
    index: dict[str, str] = {}
    for iso3, info in table().items():
        for key in (info.get("iso639_1"), info.get("bcp47"), info.get("name")):
            if key:
                index.setdefault(str(key).lower(), iso3)
    return index


def to_iso3(value: str) -> str:
    """Normalize any language code or name to ISO-639-3.

    Handles ISO-639-1 ("en"), ISO-639-3 ("eng"), fastText labels ("eng_Latn"),
    BCP-47 region/variant tags ("nb-NO", "zh-HK"), and English names. Returns
    "und" if unresolvable.
    """
    if value is None:
        return UND
    text = str(value).strip()
    if not text:
        return UND

    # fastText labels look like "eng_Latn" / "__label__eng_Latn"; keep the code.
    text = text.removeprefix("__label__").split("_", 1)[0]

    if text in table():  # already a canonical ISO-639-3 code
        return text
    hit = _reverse_index().get(text.lower())
    if hit is not None:
        return hit

    # Fallback to langcodes for tags not in the table (e.g. region subtags).
    try:
        return langcodes.Language.get(text).to_alpha3()
    except Exception:
        pass
    try:
        return langcodes.find(text).to_alpha3()
    except LookupError:
        return UND


def language_name(iso3: str) -> str:
    """English language name for an ISO-639-3 code."""
    info = table().get(iso3)
    if info and info.get("name"):
        return str(info["name"])
    try:
        return langcodes.Language.get(iso3).display_name("en")
    except Exception:
        return iso3


def iso639_1(iso3: str) -> str | None:
    """ISO-639-1 (2-letter) code for an ISO-639-3 code, or None."""
    info = table().get(iso3)
    return info.get("iso639_1") if info else None


def bcp47(iso3: str) -> str:
    """Canonical BCP-47 tag for an ISO-639-3 code."""
    info = table().get(iso3)
    return str(info["bcp47"]) if info and info.get("bcp47") else iso3
