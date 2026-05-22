"""Core metrics: macro-F1, per-language F1, confusion matrix (spec §9.1, §9.2)."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix as _sklearn_cm
from sklearn.metrics import f1_score


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Macro-averaged F1 — weights all languages equally (spec §9.1, primary metric)."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_language_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Per-language precision / recall / F1 / support."""
    labels = sorted(set(y_true) | set(y_pred))
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    rows = []
    for lang in labels:
        scores = report[lang]
        rows.append(
            {
                "lang": lang,
                "precision": scores["precision"],
                "recall": scores["recall"],
                "f1": scores["f1-score"],
                "support": int(scores["support"]),
            }
        )
    return pd.DataFrame(rows)


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str]) -> pd.DataFrame:
    """Full confusion matrix as a labeled DataFrame (rows = gold, cols = pred)."""
    labels = sorted(set(y_true) | set(y_pred))
    cm = _sklearn_cm(y_true, y_pred, labels=labels)
    return pd.DataFrame(cm, index=labels, columns=labels)
