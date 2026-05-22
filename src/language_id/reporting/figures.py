"""Figure generation (matplotlib) -> docs/figures/ (spec §12.3)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


def _read_runs_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _latest_eval_per_model_dataset(runs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        r
        for r in runs
        if (r.get("task") in {None, "eval"}) and r.get("macro_f1") is not None
    ]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("timestamp").groupby(["model", "dataset"], as_index=False).tail(1)
    return df.sort_values("macro_f1", ascending=False).reset_index(drop=True)


def _leaderboard_figure(runs: list[dict[str, Any]], out_path: Path) -> Path | None:
    df = _latest_eval_per_model_dataset(runs)
    if df.empty:
        return None
    import matplotlib.pyplot as plt

    height = max(3.0, 0.35 * len(df))
    fig, ax = plt.subplots(figsize=(8, height))
    labels = [f"{m} ({d})" for m, d in zip(df["model"], df["dataset"], strict=True)]
    ax.barh(labels[::-1], df["macro_f1"].tolist()[::-1])
    ax.set_xlabel("macro F1")
    ax.set_xlim(0, 1.0)
    ax.set_title("LID leaderboard — latest eval per (model, dataset)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".svg"))
    plt.close(fig)
    return out_path


def _copy_latest(glob_pattern: Path, dest: Path) -> Path | None:
    matches = sorted(
        glob_pattern.parent.glob(glob_pattern.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(matches[0], dest)
    svg_src = matches[0].with_suffix(".svg")
    if svg_src.exists():
        shutil.copy(svg_src, dest.with_suffix(".svg"))
    return dest


def regenerate_all(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Regenerate committed figures from artifacts under `results_dir`.

    Returns the list of files written. Missing inputs are skipped silently —
    early in a project's life many figures simply don't have data yet.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    runs = _read_runs_jsonl(Path(results_dir) / "runs.jsonl")
    leaderboard = _leaderboard_figure(runs, figures_dir / "leaderboard.png")
    if leaderboard is not None:
        written.append(leaderboard)

    experiments_dir = Path(results_dir) / "experiments"
    if experiments_dir.exists():
        for pattern, dest_name in [
            ("exp3_data_efficiency_*.png", "exp3_data_efficiency.png"),
            ("exp5_summary_*.png", "exp5_cross_benchmark.png"),
        ]:
            copied = _copy_latest(experiments_dir / pattern, figures_dir / dest_name)
            if copied is not None:
                written.append(copied)

    return written
