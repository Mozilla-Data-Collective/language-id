from collections.abc import Sequence

import pandas as pd

from language_id.codes import language_name


def accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Overall accuracy: fraction of examples where the predicted language matches the gold language."""
    if not gold:
        return 0.0
    correct = sum(g == p for g, p in zip(gold, pred, strict=True))
    return correct / len(gold)


def per_language(gold: Sequence[str], pred: Sequence[str]) -> pd.DataFrame:
    """
    Per-language metrics: support, accuracy (recall), precision, F1.
    Only languages present in the gold set are included.
    """
    df = pd.DataFrame({"gold": list(gold), "pred": list(pred)})
    labels = sorted(set(df["gold"]) | set(df["pred"]))
    rows = []
    for lang in labels:
        tp = int(((df["gold"] == lang) & (df["pred"] == lang)).sum())
        support = int((df["gold"] == lang).sum())
        n_pred = int((df["pred"] == lang).sum())
        recall = tp / support if support else 0.0
        precision = tp / n_pred if n_pred else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        rows.append(
            {
                "lang": lang,
                "name": language_name(lang),
                "support": support,
                "accuracy": recall,
                "precision": precision,
                "f1": f1,
            }
        )
    out = pd.DataFrame(rows)
    # Only languages actually present in the gold set are meaningful rows.
    return out[out["support"] > 0].reset_index(drop=True)


def macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    """Macro-averaged F1 over languages present in the gold set."""
    pl = per_language(gold, pred)
    return float(pl["f1"].mean()) if len(pl) else 0.0
