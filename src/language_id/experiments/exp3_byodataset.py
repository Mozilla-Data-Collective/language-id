"""Experiment 3 — bring-your-own-dataset OOD case study + data-efficiency curve (spec §3).

The dataset is supplied by ID in the experiment config (`dataset:`) and loaded
via `datacollective`, exactly like CommonLID / CommonVoiceLID. Marma is just the
default dataset for this experiment.

Two halves:
- Zero/few-shot LLM eval on the dataset's test split.
- Data-efficiency curve: train each of LogReg / NGramNB / XLM-R on
  n ∈ {10, 50, 100, 500, 1000} with multiple seeds; plot accuracy vs. n.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import typer

from language_id.cli import (
    _compute_metrics_bundle,
    _instantiate_llm,
    _instantiate_model,
    _load_model_cfg,
    _load_yaml,
)
from language_id.data.loaders import load_dataset_by_id, resolve_text_col
from language_id.metrics.core import macro_f1
from language_id.models.trained._sklearn_base import SklearnPipelineLIDModel

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Dataset loading / splitting
# ---------------------------------------------------------------------------


def _load_dataset_safe(dataset_id: str) -> pd.DataFrame:
    try:
        return load_dataset_by_id(dataset_id)
    except Exception as e:
        raise RuntimeError(
            f"failed to load dataset {dataset_id!r}: {e}. Confirm the `dataset:` "
            "ID in the experiment config is a valid datacollective dataset."
        ) from e


def _split_dataset(
    df: pd.DataFrame, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_pool, dev, test). Honors a `split` column if present, else 80/10/10 random."""
    if "split" in df.columns and {"train", "test"} <= set(df["split"].unique()):
        train = df[df["split"] == "train"].reset_index(drop=True)
        test = df[df["split"] == "test"].reset_index(drop=True)
        if "dev" in df["split"].unique():
            dev = df[df["split"] == "dev"].reset_index(drop=True)
        else:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(train))
            dev_n = max(50, len(train) // 10)
            dev = train.iloc[perm[:dev_n]].reset_index(drop=True)
            train = train.iloc[perm[dev_n:]].reset_index(drop=True)
        return train, dev, test
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(df))
    test_n = max(50, len(df) // 10)
    dev_n = max(50, len(df) // 10)
    test = df.iloc[perm[:test_n]].reset_index(drop=True)
    dev = df.iloc[perm[test_n : test_n + dev_n]].reset_index(drop=True)
    train = df.iloc[perm[test_n + dev_n :]].reset_index(drop=True)
    return train, dev, test


# ---------------------------------------------------------------------------
# LLM zero/few-shot eval
# ---------------------------------------------------------------------------


def _sample_few_shot(
    train_df: pd.DataFrame, n_shots: int, text_col: str, lang_col: str, seed: int
) -> list[tuple[str, str]]:
    if n_shots <= 0 or len(train_df) == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(train_df), size=min(n_shots, len(train_df)), replace=False)
    sub = train_df.iloc[idx]
    return list(zip(sub[text_col].tolist(), sub[lang_col].tolist(), strict=True))


def _run_llm_eval(
    llm_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    lang_col: str,
    out_dir: Path,
    ts: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    preds_dir = out_dir / "predictions"
    preds_dir.mkdir(parents=True, exist_ok=True)

    for model_id in llm_cfg.get("models", []):
        model_cfg = _load_model_cfg(model_id)
        for n_shots in llm_cfg.get("shots", [0]):
            shots = _sample_few_shot(train_df, n_shots, text_col, lang_col, seed=42)
            typer.echo(f"[exp3-llm] model={model_id} shots={n_shots}")
            model = _instantiate_llm(model_cfg, few_shot_examples=shots)
            preds = model.predict_batch(test_df[text_col].tolist())
            out = test_df.copy()
            out["pred"] = [p.lang_code for p in preds]
            out["confidence"] = [p.confidence for p in preds]
            out["raw_output"] = [p.raw_output for p in preds]
            metrics = _compute_metrics_bundle(out, lang_col)
            run_id = f"exp3_{model_id}_shots{n_shots}_{ts}"
            out.to_parquet(preds_dir / f"{run_id}.parquet", index=False)
            rows.append(
                {
                    "model": model_id,
                    "n_shots": n_shots,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": float((out[lang_col] == out["pred"]).mean()),
                    "n": len(out),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Data-efficiency curve
# ---------------------------------------------------------------------------


def _stratified_subsample(
    df: pd.DataFrame, n: int, seed: int, lang_col: str
) -> pd.DataFrame:
    """Take up to floor(n / n_langs) per language, then random-pad to n."""
    if len(df) <= n:
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    langs = sorted(df[lang_col].unique())
    per_lang = max(1, n // len(langs))
    parts = [
        g.sample(min(per_lang, len(g)), random_state=seed)
        for _, g in df.groupby(lang_col)
    ]
    pool = pd.concat(parts)
    if len(pool) < n:
        remainder_pool = df.drop(pool.index)
        remainder = remainder_pool.sample(
            min(n - len(pool), len(remainder_pool)), random_state=seed
        )
        pool = pd.concat([pool, remainder])
    return pool.head(n).reset_index(drop=True)


def _fit_and_eval(
    model: Any,
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    lang_col: str,
) -> tuple[float, float]:
    if isinstance(model, SklearnPipelineLIDModel):
        model.fit(train_df, text_col=text_col, lang_col=lang_col)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            model.fit(
                train_df=train_df,
                eval_df=dev_df,
                text_col=text_col,
                lang_col=lang_col,
                output_dir=Path(tmp),
            )
    preds = model.predict_batch(test_df[text_col].tolist())
    pred_langs = [p.lang_code for p in preds]
    gold = test_df[lang_col].tolist()
    acc = float(sum(g == p for g, p in zip(gold, pred_langs)) / max(1, len(gold)))
    f1 = macro_f1(gold, pred_langs)
    return acc, f1


def _run_data_efficiency(
    de_cfg: dict[str, Any],
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str,
    lang_col: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id in de_cfg.get("models", []):
        model_cfg = _load_model_cfg(model_id)
        for n in de_cfg.get("ns", []):
            for seed in de_cfg.get("seeds", [42]):
                sub = _stratified_subsample(train_df, n=n, seed=seed, lang_col=lang_col)
                if len(sub) == 0:
                    continue
                typer.echo(
                    f"[exp3-de] model={model_id} n={n} seed={seed} (train_rows={len(sub)})"
                )
                model = _instantiate_model(model_cfg)
                acc, f1 = _fit_and_eval(
                    model, sub, dev_df, test_df, text_col, lang_col
                )
                rows.append(
                    {
                        "model": model_id,
                        "n": int(n),
                        "seed": int(seed),
                        "accuracy": acc,
                        "macro_f1": f1,
                    }
                )
    return pd.DataFrame(rows)


def _plot_curve(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for model_id, group in df.groupby("model"):
        agg = group.groupby("n")["accuracy"].agg(["mean", "std"]).reset_index()
        ax.errorbar(
            agg["n"],
            agg["mean"],
            yerr=agg["std"].fillna(0),
            label=model_id,
            marker="o",
            capsize=3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("training examples (n)")
    ax.set_ylabel("accuracy on test")
    ax.set_title("Data-efficiency curve (mean ± std across seeds)")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(config_path: Path) -> None:
    cfg = _load_yaml(Path(config_path))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _REPO_ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_id = cfg.get("dataset")
    if not dataset_id:
        raise RuntimeError(f"config {config_path} has no `dataset:` ID")

    df = _load_dataset_safe(dataset_id)
    text_col = resolve_text_col(df)
    lang_col = "lang"
    train_pool, dev_df, test_df = _split_dataset(df)
    typer.echo(
        f"[exp3] dataset={dataset_id} train={len(train_pool)} dev={len(dev_df)} "
        f"test={len(test_df)} (text_col={text_col!r})"
    )

    if "llm_eval" in cfg:
        llm_summary = _run_llm_eval(
            cfg["llm_eval"], train_pool, test_df, text_col, lang_col, out_dir, ts
        )
        llm_path = out_dir / f"exp3_llm_few_shot_{ts}.parquet"
        llm_summary.to_parquet(llm_path, index=False)
        typer.echo(f"[exp3] LLM summary -> {llm_path}")

    if "data_efficiency" in cfg:
        de_df = _run_data_efficiency(
            cfg["data_efficiency"], train_pool, dev_df, test_df, text_col, lang_col
        )
        de_path = out_dir / f"exp3_data_efficiency_{ts}.parquet"
        de_df.to_parquet(de_path, index=False)
        plot_path = out_dir / f"exp3_data_efficiency_{ts}.png"
        _plot_curve(de_df, plot_path)
        typer.echo(f"[exp3] data-efficiency parquet -> {de_path}")
        typer.echo(f"[exp3] data-efficiency plot    -> {plot_path}")


if __name__ == "__main__":
    typer.run(run)
