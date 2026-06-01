from typing import Any

import pandas as pd

from language_id import metrics
from language_id.lang_codes_mapping import to_iso3
from language_id.data import TEXT_COLUMN_NAME
from language_id.models import LIDModel


def evaluate(df: pd.DataFrame, model: LIDModel) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Evaluate `model` on `df` (which must have a `lang` gold column).

    Returns (overall, per_language_df, predictions_df). Predictions are
    normalized to ISO-639-3 before scoring.
    """
    preds = model.predict_batch(df[TEXT_COLUMN_NAME].tolist())

    predictions = df.copy()
    predictions["pred"] = [to_iso3(p.lang_code) for p in preds]
    predictions["confidence"] = [p.confidence for p in preds]
    predictions["raw_output"] = [p.raw_output for p in preds]

    gold = predictions["lang"].tolist()
    pred = predictions["pred"].tolist()
    overall = {
        "model": getattr(model, "name", type(model).__name__),
        "n": len(predictions),
        "n_languages": predictions["lang"].nunique(),
        "accuracy": metrics.accuracy(gold, pred),
        "macro_f1": metrics.macro_f1(gold, pred),
    }
    return overall, metrics.per_language(gold, pred), predictions
