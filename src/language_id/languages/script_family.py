"""ISO 15924 script grouping for per-script-family metric slicing (spec §9.2)."""

from __future__ import annotations

from langcodes import Language

from language_id.languages.codes import to_bcp47

_UNKNOWN_SCRIPT = "Zzzz"


def script_for(lang_code: str) -> str:
    """Return the dominant ISO 15924 script code for a BCP-47 tag (e.g. "Latn", "Cyrl", "Hans")."""
    lang = Language.get(to_bcp47(lang_code)).maximize()
    return lang.script or _UNKNOWN_SCRIPT


def script_family(lang_code: str) -> str:
    """Return a coarse script-family label used for slicing metrics.

    Hans / Hant are unified to the parent Hani script for coarse-family grouping.
    """
    script = script_for(lang_code)
    if script in {"Hans", "Hant"}:
        return "Hani"
    return script
