"""Confusable-pair accuracy (spec §9.2)."""

from __future__ import annotations

import pandas as pd


def confusable_pair_accuracy(
    predictions: pd.DataFrame,
    pairs: list[tuple[str, str]],
    gold_col: str = "lang",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Per-pair accuracy restricted to rows where the gold label is in the pair."""
    raise NotImplementedError("confusable_pair_accuracy not implemented yet")
