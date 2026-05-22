"""Experiment 5 — Cross-benchmark evaluation (CommonVoiceLID-trained classifiers on CommonLID test) (spec §3).

Loads the latest train-run checkpoint for each model from `results/runs.jsonl`
(or an explicit `checkpoints:` override block in the experiment YAML) and runs
`_eval_impl` against CommonLID under the same sampling block.

Output: `results/experiments/exp5_summary.parquet` with one row per model.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer

from language_id.cli import _eval_impl, _load_yaml

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _latest_checkpoint(runs_jsonl: Path, model_id: str) -> Path:
    if not runs_jsonl.exists():
        raise RuntimeError(
            f"no runs registry at {runs_jsonl}; train models first via "
            "`language-id train --model <id> --dataset commonvoice_lid`"
        )
    candidates: list[dict[str, Any]] = []
    for line in runs_jsonl.read_text().splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("model") != model_id:
            continue
        if rec.get("task") and rec["task"] != "train":
            continue
        if not rec.get("artifact_path"):
            continue
        candidates.append(rec)
    if not candidates:
        raise RuntimeError(
            f"no training run found for model={model_id!r} in {runs_jsonl}"
        )
    candidates.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return _REPO_ROOT / candidates[0]["artifact_path"]


def run(config_path: Path) -> None:
    """Load trained classifiers, evaluate on CommonLID, write a summary parquet."""
    cfg = _load_yaml(Path(config_path))
    runs_jsonl = _REPO_ROOT / "results" / "runs.jsonl"
    explicit = cfg.get("checkpoints") or {}
    dataset_id = cfg.get("dataset", "commonlid")

    summary_rows: list[dict[str, Any]] = []
    for model_id in cfg["models"]:
        if model_id in explicit:
            ckpt = Path(explicit[model_id])
            if not ckpt.is_absolute():
                ckpt = _REPO_ROOT / ckpt
        else:
            ckpt = _latest_checkpoint(runs_jsonl, model_id)

        typer.echo(f"[exp5] model={model_id} checkpoint={ckpt}")
        _eval_impl(
            model_id=model_id,
            dataset_id=dataset_id,
            config_path=Path(config_path),
            limit=None,
            checkpoint=ckpt,
        )

    # Re-read the just-written entries from runs.jsonl to build the summary.
    if runs_jsonl.exists():
        for line in runs_jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                rec.get("task") == "eval"
                and rec.get("dataset") == dataset_id
                and rec.get("model") in cfg["models"]
            ):
                summary_rows.append(
                    {
                        "model": rec["model"],
                        "macro_f1": rec.get("macro_f1"),
                        "n": rec.get("n"),
                        "run_id": rec.get("run_id"),
                        "predictions_path": rec.get("predictions_path"),
                    }
                )

    # Deduplicate to keep only the latest entry per model (rows are in append order).
    latest_by_model: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        latest_by_model[row["model"]] = row
    summary_df = pd.DataFrame(latest_by_model.values())

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = _REPO_ROOT / "results" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"exp5_summary_{ts}.parquet"
    summary_df.to_parquet(out_path, index=False)

    typer.echo("")
    typer.echo(f"[exp5] summary -> {out_path}")
    for row in summary_df.itertuples(index=False):
        typer.echo(f"  {row.model:20s} macro_F1={row.macro_f1:.4f}  n={row.n}")


if __name__ == "__main__":
    typer.run(run)
