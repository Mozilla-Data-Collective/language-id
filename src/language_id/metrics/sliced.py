"""Sliced metrics: script family, resource tier, length bucket (spec §9.2, §7.4)."""

from __future__ import annotations

import pandas as pd

from language_id.languages.resource_tier import tier_for
from language_id.languages.script_family import script_family
from language_id.metrics.core import macro_f1


def _slice_macro_f1(
    predictions: pd.DataFrame,
    slice_col: str,
    gold_col: str,
    pred_col: str,
) -> pd.DataFrame:
    rows = []
    for value, sub in predictions.groupby(slice_col, dropna=False):
        rows.append(
            {
                slice_col: value,
                "accuracy": float((sub[gold_col] == sub[pred_col]).mean()),
                "macro_f1": macro_f1(sub[gold_col].tolist(), sub[pred_col].tolist()),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def by_length_bucket(
    predictions: pd.DataFrame,
    gold_col: str = "lang",
    pred_col: str = "pred",
    bucket_col: str = "length_bucket",
) -> pd.DataFrame:
    """Accuracy + macro-F1 per length bucket. Uses the SAME bucket boundaries as sampling (spec §7.4)."""
    return _slice_macro_f1(predictions, bucket_col, gold_col, pred_col)


def by_script_family(
    predictions: pd.DataFrame,
    gold_col: str = "lang",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Accuracy + macro-F1 per ISO 15924 script-family grouping."""
    preds = predictions.copy()
    preds["script_family"] = preds[gold_col].map(script_family)
    return _slice_macro_f1(preds, "script_family", gold_col, pred_col)


def by_resource_tier(
    predictions: pd.DataFrame,
    gold_col: str = "lang",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Accuracy + macro-F1 per resource tier (high / mid / low / unknown)."""
    preds = predictions.copy()
    preds["tier"] = preds[gold_col].map(tier_for)
    return _slice_macro_f1(preds, "tier", gold_col, pred_col)
