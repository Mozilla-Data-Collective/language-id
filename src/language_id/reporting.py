"""Persist an evaluation run: predictions, results, metrics, and graphs.

Each call to `save_run` creates a timestamped directory under `results/` holding
everything produced by one evaluation, plus appends a one-line summary to
`results/runs.jsonl` for cross-run comparison.

    results/<model>_<dataset>_<timestamp>/
        predictions.csv      every row with gold, pred, confidence, raw_output
        per_language.csv      per-language support / accuracy / precision / f1
        metrics.json          overall results + per-language metrics
        plots/per_language_f1.png
        plots/per_language_metrics.png
        plots/confusion_matrix.png
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: never try to open a window
import matplotlib.pyplot as plt
import pandas as pd


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def save_run(
    out_root: Path,
    model: str,
    dataset: str,
    overall: dict[str, Any],
    per_lang: pd.DataFrame,
    predictions: pd.DataFrame,
) -> Path:
    """Write predictions, metrics, and graphs for one run and return the run directory."""
    timestamp = _timestamp()
    stem = f"{model}_{dataset.replace('/', '_')}_{timestamp}"
    run_dir = out_root / stem
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Predictions and per-language metrics.
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    per_lang.to_csv(run_dir / "per_language.csv", index=False)

    # Overall results + per-language metrics in one JSON.
    metrics = {
        **overall,
        "dataset": dataset,
        "timestamp": timestamp,
        "per_language": per_lang.to_dict(orient="records"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    _plot_per_language_f1(per_lang, plots_dir / "per_language_f1.png")
    _plot_per_language_metrics(per_lang, plots_dir / "per_language_metrics.png")
    _plot_confusion_matrix(predictions, plots_dir / "confusion_matrix.png")

    # Append a one-line summary for cross-run comparison.
    summary = {
        "run_id": stem,
        "model": model,
        "dataset": dataset,
        "timestamp": timestamp,
        "accuracy": overall.get("accuracy"),
        "macro_f1": overall.get("macro_f1"),
        "n": overall.get("n"),
        "n_languages": overall.get("n_languages"),
    }
    with (out_root / "runs.jsonl").open("a") as f:
        f.write(json.dumps(summary) + "\n")

    return run_dir


def _label(per_lang: pd.DataFrame) -> pd.Series:
    """Readable per-language axis labels: "name (code)" when a name exists."""
    if "name" in per_lang.columns:
        return per_lang.apply(
            lambda r: f"{r['name']} ({r['lang']})" if pd.notna(r["name"]) else str(r["lang"]),
            axis=1,
        )
    return per_lang["lang"].astype(str)


def _plot_per_language_f1(per_lang: pd.DataFrame, path: Path) -> None:
    if per_lang.empty:
        return
    df = per_lang.sort_values("f1", ascending=True)
    labels = _label(df)
    height = max(3.0, 0.3 * len(df))
    fig, ax = plt.subplots(figsize=(8, height))
    ax.barh(labels, df["f1"], color="#4C72B0")
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("Per-language F1")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_per_language_metrics(per_lang: pd.DataFrame, path: Path) -> None:
    if per_lang.empty:
        return
    df = per_lang.sort_values("f1", ascending=True)
    labels = _label(df)
    y = range(len(df))
    height = max(3.0, 0.4 * len(df))
    fig, ax = plt.subplots(figsize=(9, height))
    bar_h = 0.27
    ax.barh([i + bar_h for i in y], df["accuracy"], bar_h, label="accuracy", color="#55A868")
    ax.barh(list(y), df["precision"], bar_h, label="precision", color="#C44E52")
    ax.barh([i - bar_h for i in y], df["f1"], bar_h, label="f1", color="#4C72B0")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1)
    ax.set_xlabel("score")
    ax.set_title("Per-language accuracy / precision / F1")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_confusion_matrix(predictions: pd.DataFrame, path: Path) -> None:
    """Confusion matrix over the languages present in the gold set."""
    if predictions.empty or "lang" not in predictions or "pred" not in predictions:
        return
    gold_labels = sorted(predictions["lang"].dropna().unique())
    if not gold_labels:
        return
    # Columns = gold labels plus any extra languages the model predicted.
    pred_only = sorted(set(predictions["pred"].dropna()) - set(gold_labels))
    col_labels = gold_labels + pred_only
    cm = pd.crosstab(predictions["lang"], predictions["pred"]).reindex(
        index=gold_labels, columns=col_labels, fill_value=0
    )

    size = max(6.0, 0.4 * len(col_labels))
    fig, ax = plt.subplots(figsize=(size, max(6.0, 0.4 * len(gold_labels))))
    im = ax.imshow(cm.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(gold_labels)))
    ax.set_yticklabels(gold_labels, fontsize=7)
    ax.set_xlabel("predicted")
    ax.set_ylabel("gold")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)