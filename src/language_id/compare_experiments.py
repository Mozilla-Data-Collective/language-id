"""
Compare evaluation runs that were already saved under ``results/``.

`save_run` writes one timestamped directory per evaluation:
results/<model>_<dataset>_<timestamp>/
    metrics.json      overall scores + per-language metrics
    per_language.csv  per-language support / recall / precision / f1
    predictions.csv   every row with gold, pred, confidence

This module reads those directories back so several models evaluated on the
**same dataset / experiment setup** can be compared side by side, which model
won overall, and where each one wins or loses per language.

Typical use (e.g. from a notebook)::

    from language_id.compare_experiments import load_runs, overview_table, per_language_pivot

    runs = load_runs(["glotlid_commonlid_...", "langdetect_commonlid_...", "nllb-lid_commonlid_..."])
    overview_table(runs)              # one row per model, sorted by macro-F1
    per_language_pivot(runs, "f1")    # languages x models

Only runs that share the same dataset and the same per-language support are
directly comparable; :func:`check_comparable` flags any that are not.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


@dataclass
class Run:
    """One saved evaluation run loaded back from disk."""

    run_id: str
    model: str
    dataset: str
    timestamp: str
    accuracy: float
    macro_f1: float
    n: int
    n_languages: int
    per_language: pd.DataFrame
    run_dir: Path

    @property
    def label(self) -> str:
        """Short, human-friendly column label for tables and plots."""
        return self.model


def discover_runs(results_root: Path | str = RESULTS_ROOT) -> list[str]:
    """Return the run ids (directory names) of every saved run, newest first."""
    root = Path(results_root)
    run_ids = [p.name for p in root.iterdir() if p.is_dir() and (p / "metrics.json").exists()]
    return sorted(run_ids, reverse=True)


def load_run(run_id: str, results_root: Path | str = RESULTS_ROOT) -> Run:
    """Load a single saved run by its directory name (run id)."""
    run_dir = Path(results_root) / run_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics.json in {run_dir} — is '{run_id}' a valid run id?")
    metrics = pd.read_json(metrics_path, typ="series")
    per_language = pd.read_csv(run_dir / "per_language.csv")
    return Run(
        run_id=run_id,
        model=str(metrics.get("model", run_id.split("_")[0])),
        dataset=str(metrics.get("dataset", "")),
        timestamp=str(metrics.get("timestamp", "")),
        accuracy=float(metrics.get("accuracy", float("nan"))),
        macro_f1=float(metrics.get("macro_f1", float("nan"))),
        n=int(metrics.get("n", len(per_language))),
        n_languages=int(metrics.get("n_languages", per_language["lang"].nunique())),
        per_language=per_language,
        run_dir=run_dir,
    )


def load_runs(
    run_ids: list[str] | None = None,
    results_root: Path | str = RESULTS_ROOT,
    dataset: str | None = None,
) -> list[Run]:
    """Load several runs.

    With no ``run_ids`` every saved run is loaded; pass ``dataset`` to keep only
    runs on that dataset. Results are sorted by macro-F1, best first.
    """
    if run_ids is None:
        run_ids = discover_runs(results_root)
    runs = [load_run(r, results_root) for r in run_ids]
    if dataset is not None:
        runs = [r for r in runs if r.dataset == dataset]
    return sorted(runs, key=lambda r: r.macro_f1 if pd.notna(r.macro_f1) else -1, reverse=True)


def check_comparable(runs: list[Run]) -> list[str]:
    """Return human-readable warnings if the runs are not directly comparable.

    Runs are directly comparable only when they share a dataset and were scored
    on the same per-language support (i.e. the same sampled rows). An empty list
    means the comparison is clean.
    """
    warnings: list[str] = []
    if len(runs) < 2:
        return warnings

    datasets = {r.dataset for r in runs}
    if len(datasets) > 1:
        warnings.append(
            f"Runs span different datasets: {sorted(datasets)} — not directly comparable."
        )

    ns = {r.n for r in runs}
    if len(ns) > 1:
        warnings.append(f"Runs cover different sample counts (n): {sorted(ns)}.")

    # Same languages and same support per language -> same evaluation rows.
    supports = [r.per_language.set_index("lang")["support"].sort_index() for r in runs]
    ref = supports[0]
    for run, sup in zip(runs[1:], supports[1:], strict=True):
        if not ref.index.equals(sup.index):
            warnings.append(
                f"'{run.label}' covers a different set of languages than '{runs[0].label}'."
            )
        elif not ref.equals(sup):
            warnings.append(
                f"'{run.label}' has different per-language support than '{runs[0].label}'."
            )
    return warnings


def overview_table(runs: list[Run]) -> pd.DataFrame:
    """One row per run: model, dataset, n, n_languages, accuracy, macro_f1 (best first)."""
    rows = [
        {
            "model": r.label,
            "dataset": r.dataset,
            "n": r.n,
            "n_languages": r.n_languages,
            "accuracy": r.accuracy,
            "macro_f1": r.macro_f1,
            "run_id": r.run_id,
        }
        for r in runs
    ]
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)


def per_language_pivot(runs: list[Run], metric: str = "f1") -> pd.DataFrame:
    """Per-language ``metric`` for every run: rows are languages, columns are models.

    A leading ``support`` column carries the gold-sample count per language and
    rows are sorted by support (most-represented languages first).
    """
    frames = []
    for r in runs:
        f = r.per_language[["lang", "name", "support", metric]].copy()
        f["model"] = r.label
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)

    pivot = long.pivot_table(index=["name", "lang"], columns="model", values=metric)
    support = long.groupby(["name", "lang"])["support"].max()
    pivot.insert(0, "support", support)
    return pivot.sort_values("support", ascending=False)


def style_overview(table: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Color-graded view of :func:`overview_table` for notebook display."""
    metric_cols = [c for c in ("accuracy", "macro_f1") if c in table]
    return table.style.format({c: "{:.3f}" for c in metric_cols}).background_gradient(
        subset=metric_cols, cmap="Greens", vmin=0, vmax=1
    )


def style_per_language(pivot: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Color-graded view of :func:`per_language_pivot` for notebook display."""
    model_cols = [c for c in pivot.columns if c != "support"]
    return pivot.style.format(
        {**{m: "{:.2f}" for m in model_cols}, "support": "{:.0f}"}
    ).background_gradient(subset=model_cols, cmap="RdYlGn", vmin=0, vmax=1)


def plot_overview(runs: list[Run], ax: plt.Axes | None = None) -> plt.Axes:
    """Grouped bar chart of accuracy and macro-F1 per model, best macro-F1 first."""
    table = overview_table(runs).set_index("model")[["accuracy", "macro_f1"]]
    if ax is None:
        _, ax = plt.subplots(figsize=(max(6.0, 1.4 * len(table)), 4))
    table.plot.bar(ax=ax, rot=20, ylim=(0, 1), color=["#4C72B0", "#55A868"])
    ax.set_ylabel("score")
    ax.set_title("Model comparison — overall")
    ax.legend(title="metric")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=7, padding=2)
    return ax


def _heatmap_figsize(n_langs: int, n_models: int, lang_axis: str) -> tuple[float, float]:
    """Figure size scaled to the grid, with languages along ``lang_axis``."""
    lang_extent = max(4.0, 0.32 * n_langs)
    model_extent = max(3.0, 0.45 * n_models)
    return (model_extent, lang_extent) if lang_axis == "y" else (lang_extent, model_extent)


def _draw_heatmap_panel(
    hm: pd.DataFrame, metric: str, ax: plt.Axes, title: str, lang_axis: str = "y"
) -> None:
    """Draw one heatmap panel on ``ax``.

    ``hm`` must have languages on the rows ((name, code) tuples) and models on
    the columns; ``lang_axis="x"`` draws its transpose.
    """
    lang_labels = [f"{name} ({code})" for name, code in hm.index]
    model_labels = list(hm.columns)
    values = hm.values
    if lang_axis == "x":
        values = values.T
        x_labels, y_labels = lang_labels, model_labels
        lang_ticks = ax.xaxis
    else:
        x_labels, y_labels = model_labels, lang_labels
        lang_ticks = ax.yaxis

    im = ax.imshow(values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(x_labels)), x_labels, rotation=90)
    ax.set_yticks(range(len(y_labels)), y_labels)
    lang_ticks.set_tick_params(labelsize=7)
    ax.set_title(title)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=metric)


def plot_per_language_heatmap(
    runs: list[Run],
    metric: str = "f1",
    ax: plt.Axes | None = None,
    langs_per_fig: int | None = None,
    lang_axis: str = "y",
) -> plt.Axes | list[plt.Axes]:
    """Heatmap of per-language ``metric``.

    ``lang_axis`` selects which axis holds the languages: ``"y"`` (default)
    puts languages on the rows and models on the columns, ``"x"`` the
    transpose. With ``langs_per_fig`` set, the languages are split into
    chunks of that size, one figure is drawn per chunk and a list of axes
    is returned.
    """
    if lang_axis not in ("x", "y"):
        raise ValueError("lang_axis must be 'x' or 'y'.")
    pivot = per_language_pivot(runs, metric)
    model_cols = [c for c in pivot.columns if c != "support"]
    hm = pivot[model_cols]  # languages on the rows; transposed at draw time
    n_langs = len(hm)

    if langs_per_fig is None:
        if ax is None:
            _, ax = plt.subplots(figsize=_heatmap_figsize(n_langs, len(model_cols), lang_axis))
        _draw_heatmap_panel(hm, metric, ax, f"Per-language {metric}", lang_axis)
        return ax

    axes = []
    for start in range(0, n_langs, langs_per_fig):
        chunk = hm.iloc[start : start + langs_per_fig]
        _, cax = plt.subplots(figsize=_heatmap_figsize(len(chunk), len(model_cols), lang_axis))
        title = f"Per-language {metric} ({start + 1}-{start + len(chunk)} of {n_langs})"
        _draw_heatmap_panel(chunk, metric, cax, title, lang_axis)
        axes.append(cax)
    return axes


def plot_disagreement(
    runs: list[Run], metric: str = "f1", top: int = 15, ax: plt.Axes | None = None
) -> plt.Axes:
    """Languages with the largest best-minus-worst spread across models."""
    pivot = per_language_pivot(runs, metric)
    model_cols = [c for c in pivot.columns if c != "support"]
    if len(model_cols) < 2:
        raise ValueError("Need at least two runs to measure disagreement.")
    hm = pivot[model_cols]
    spread = (hm.max(axis=1) - hm.min(axis=1)).sort_values(ascending=False)
    sel = spread.head(top).index[::-1]
    top_df = hm.loc[sel]
    if ax is None:
        _, ax = plt.subplots(figsize=(9, max(4.0, 0.5 * len(top_df))))
    top_df.plot.barh(ax=ax, xlim=(0, 1))
    ax.set_yticklabels([f"{n} ({code})" for n, code in top_df.index])
    ax.set_xlabel(metric)
    ax.set_title(f"Top {len(top_df)} languages by cross-model {metric} spread")
    ax.legend(title="model", loc="lower right")
    return ax
