"""Known confusable language pairs (spec §9.2). Extend as confusion-matrix analysis reveals more."""

from __future__ import annotations

# Pairs are unordered; treat (a, b) == (b, a) when scoring confusable-pair accuracy.
DEFAULT_CONFUSABLE_PAIRS: list[tuple[str, str]] = [
    ("hi", "ur"),    # Hindi / Urdu
    ("sr", "hr"),    # Serbian / Croatian
    ("sr", "bs"),    # Serbian / Bosnian
    ("hr", "bs"),    # Croatian / Bosnian
    ("id", "ms"),    # Indonesian / Malay
    ("nb", "nn"),    # Norwegian Bokmål / Nynorsk
]
