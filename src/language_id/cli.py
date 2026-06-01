"""Typer CLI entry point.

Flat command structure per spec §10:
    language-id eval         --model <id> --dataset <id> [--config <path>] [--limit N] [--checkpoint <dir>]
    language-id train        --model <id> --dataset <id> [--config <path>] [--limit N]
    language-id add-language --dataset <id> [--config <path>]
    language-id report       [--from results/] [--to docs/]
    language-id compute-tiers
"""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import yaml

from language_id.data.loaders import (
    load_commonlid,
    load_commonvoice_lid,
    load_dataset_by_id,
    load_marma,
    resolve_text_col,
)
from language_id.data.sampling import SamplingConfig, length_stratified_sample
from language_id.metrics.core import macro_f1, per_language_f1
from language_id.metrics.sliced import (
    by_length_bucket,
    by_resource_tier,
    by_script_family,
)

app = typer.Typer(
    name="language-id",
    help="Benchmarking and building text language identification.",
    no_args_is_help=True,
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def eval(
    model: str = typer.Option(..., "--model", help="Model id (filename stem in configs/models/)."),
    dataset: str = typer.Option(..., "--dataset", help="Dataset id."),
    config: Path | None = typer.Option(None, "--config", help="Experiment YAML."),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of examples after sampling."),
    checkpoint: Path | None = typer.Option(
        None, "--checkpoint", help="Trained-model artifact directory (required for kind=trained)."
    ),
) -> None:
    """Run a single (model, dataset) evaluation."""
    _eval_impl(
        model_id=model,
        dataset_id=dataset,
        config_path=config,
        limit=limit,
        checkpoint=checkpoint,
    )


@app.command()
def train(
    model: str = typer.Option(..., "--model", help="Model id (must be of kind=trained)."),
    dataset: str = typer.Option(..., "--dataset", help="Dataset id (commonvoice_lid / marma)."),
    config: Path | None = typer.Option(None, "--config", help="Experiment YAML."),
    limit: int | None = typer.Option(None, "--limit", help="Limit train rows (for quick iteration)."),
) -> None:
    """Train a classifier, save artifacts to results/models/<run_id>/."""
    _train_impl(model_id=model, dataset_id=dataset, config_path=config, limit=limit)


@app.command("add-language")
def add_language(
    dataset: str = typer.Option(
        ..., "--dataset", help="Mozilla Data Collective dataset ID for your corpus."
    ),
    config: Path | None = typer.Option(
        None,
        "--config",
        help="Experiment YAML (defaults to configs/experiments/exp3_byodataset.yaml).",
    ),
) -> None:
    """Experiment 4: data-efficiency curve on a user-supplied datacollective dataset."""
    _add_language_impl(dataset_id=dataset, config_path=config)


@app.command()
def report(
    from_dir: Path = typer.Option(Path("results"), "--from", help="Results directory."),
    to_dir: Path = typer.Option(Path("docs"), "--to", help="Docs directory."),
) -> None:
    """Regenerate figures and markdown tables from results/ into docs/."""
    from language_id.reporting.figures import regenerate_all as _regen_figures
    from language_id.reporting.tables import regenerate_all as _regen_tables

    results_dir = from_dir if from_dir.is_absolute() else _REPO_ROOT / from_dir
    docs_dir = to_dir if to_dir.is_absolute() else _REPO_ROOT / to_dir
    figures_dir = docs_dir / "figures"

    typer.echo(f"Figures -> {figures_dir}")
    figs = _regen_figures(results_dir, figures_dir)
    for f in figs:
        typer.echo(f"  wrote {f.relative_to(_REPO_ROOT)}")

    typer.echo(f"Tables  -> {docs_dir}")
    pages = _regen_tables(results_dir, docs_dir)
    for p in pages:
        typer.echo(f"  wrote {p.relative_to(_REPO_ROOT)}")


@app.command("compute-tiers")
def compute_tiers() -> None:
    """Regenerate per-language resource-tier JSON from CommonVoiceLID line counts (spec §11)."""
    from language_id.languages.resource_tier import compute_tiers_from_counts

    cfg_path = _REPO_ROOT / "configs" / "resource_tiers.yaml"
    cfg = _load_yaml(cfg_path)
    thresholds = cfg["thresholds"]
    out_rel = cfg.get("output_path") or "src/language_id/languages/_tiers.json"
    output_path = Path(out_rel)
    if not output_path.is_absolute():
        output_path = _REPO_ROOT / output_path

    typer.echo("Counting CommonVoiceLID train rows per language…")
    df = load_commonvoice_lid(split="train")
    counts = df.groupby("lang").size().astype(int).to_dict()
    tiers = compute_tiers_from_counts(counts, thresholds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(tiers, indent=2, sort_keys=True))
    by_tier: dict[str, int] = {}
    for t in tiers.values():
        by_tier[t] = by_tier.get(t, 0) + 1
    typer.echo(f"Wrote {len(tiers)} entries -> {output_path.relative_to(_REPO_ROOT)}")
    for tier, n in sorted(by_tier.items()):
        typer.echo(f"  {tier:8s} {n}")


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RESERVED_CFG_KEYS = {"name", "kind", "implementation", "version", "options"}


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def _load_model_cfg(model_id: str) -> dict[str, Any]:
    path = _REPO_ROOT / "configs" / "models" / f"{model_id}.yaml"
    if not path.exists():
        raise typer.BadParameter(f"model config not found: {path}")
    return _load_yaml(path)


def _model_class(impl: str) -> Any:
    if not impl or ":" not in impl:
        raise typer.BadParameter(
            f"model config missing 'implementation: module:Class' (got {impl!r})"
        )
    module_name, class_name = impl.split(":", 1)
    return getattr(importlib.import_module(module_name), class_name)


def _resolve_dataset(
    dataset_id: str,
    exp_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, str, str]:
    """Return (df, text_col, lang_col) for the eval path."""
    if dataset_id == "commonlid":
        # TODO: confirm text column name for CommonLID; "text" is the obvious default.
        return load_commonlid(), "text", "lang"
    if dataset_id == "commonvoice_lid":
        split = exp_cfg.get("split") or exp_cfg.get("splits", {}).get("test") or "test"
        return load_commonvoice_lid(split=split), "sentence", "lang"
    if dataset_id == "marma":
        return load_marma(), "text", "lang"
    # Bring-your-own-data: treat any other id as a datacollective dataset ID.
    df = load_dataset_by_id(dataset_id)
    return df, resolve_text_col(df), "lang"


def _resolve_train_dataset(
    dataset_id: str,
    exp_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Return (train_df, eval_df, text_col, lang_col) for the train path."""
    if dataset_id == "commonvoice_lid":
        splits = exp_cfg.get("splits") or {}
        train_split = splits.get("train", "train")
        eval_split = splits.get("dev", "dev")
        train = load_commonvoice_lid(split=train_split)
        dev = load_commonvoice_lid(split=eval_split)
        return train, dev, "sentence", "lang"
    if dataset_id == "marma":
        df = load_marma()
        if "split" in df.columns:
            return (
                df[df["split"] == "train"].reset_index(drop=True),
                df[df["split"] == "dev"].reset_index(drop=True),
                "text",
                "lang",
            )
        # No splits: caller responsibility (e.g. add-language workflow handles this).
        raise typer.BadParameter("marma loader returned no `split` column; provide one or use add-language")
    raise typer.BadParameter(
        f"dataset {dataset_id!r} cannot be used for training (no train/dev splits)."
    )


def _instantiate_model(model_cfg: dict[str, Any]) -> Any:
    kind = model_cfg.get("kind")
    if kind in {"standard", "trained"}:
        cls = _model_class(model_cfg.get("implementation", ""))
        extras = {k: v for k, v in model_cfg.items() if k not in _RESERVED_CFG_KEYS}
        return cls(**extras, **(model_cfg.get("options") or {}))
    if kind == "llm":
        return _instantiate_llm(model_cfg)
    raise typer.BadParameter(f"unsupported model kind: {kind!r}")


def _instantiate_llm(
    model_cfg: dict[str, Any],
    few_shot_examples: list[tuple[str, str]] | None = None,
) -> Any:
    provider = model_cfg["provider"]
    if provider == "together":
        from language_id.models.llm.together_client import TogetherModel

        cls: Any = TogetherModel
    else:
        from language_id.models.llm.any_llm_client import AnyLLMModel

        cls = AnyLLMModel

    return cls(
        name=model_cfg["name"],
        version=str(model_cfg.get("version", "TBD")),
        provider=provider,
        model_id=model_cfg["model_id"],
        prompt=model_cfg["prompt"],
        client_options=model_cfg.get("client") or {},
        few_shot_examples=few_shot_examples,
    )


def _config_hash(*configs: dict[str, Any]) -> str:
    blob = json.dumps(configs, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return out.stdout.strip() or None


def _compute_metrics_bundle(out: pd.DataFrame, lang_col: str) -> dict[str, Any]:
    macro = macro_f1(out[lang_col].tolist(), out["pred"].tolist())
    metrics: dict[str, Any] = {
        "macro_f1": macro,
        "n": int(len(out)),
        "n_languages": int(out[lang_col].nunique()),
        "per_language": per_language_f1(
            out[lang_col].tolist(), out["pred"].tolist()
        ).to_dict(orient="records"),
        "by_script_family": by_script_family(out, gold_col=lang_col).to_dict(orient="records"),
        "by_resource_tier": by_resource_tier(out, gold_col=lang_col).to_dict(orient="records"),
    }
    if "length_bucket" in out.columns:
        metrics["by_length_bucket"] = by_length_bucket(out, gold_col=lang_col).to_dict(
            orient="records"
        )
    return metrics


def _eval_impl(
    model_id: str,
    dataset_id: str,
    config_path: Path | None,
    limit: int | None,
    checkpoint: Path | None,
) -> None:
    results_dir = _REPO_ROOT / "results"
    model_cfg = _load_model_cfg(model_id)

    exp_cfg: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.exists():
            raise typer.BadParameter(f"experiment config not found: {config_path}")
        exp_cfg = _load_yaml(config_path)

    typer.echo(f"Loading dataset: {dataset_id}")
    df, text_col, lang_col = _resolve_dataset(dataset_id, exp_cfg)

    if "sampling" in exp_cfg:
        typer.echo("Applying length-stratified sampling…")
        sampling = SamplingConfig(**exp_cfg["sampling"])
        df = length_stratified_sample(df, text_col=text_col, lang_col=lang_col, config=sampling)
        typer.echo(f"  sampled {len(df)} rows across {df[lang_col].nunique()} languages")

    if limit is not None:
        df = df.head(limit).reset_index(drop=True)
        typer.echo(f"  --limit {limit}: kept {len(df)} rows")

    kind = model_cfg.get("kind")
    if kind == "trained":
        if checkpoint is None:
            raise typer.BadParameter(
                f"--checkpoint is required when evaluating kind=trained model {model_id!r}"
            )
        typer.echo(f"Loading trained checkpoint: {checkpoint}")
        cls = _model_class(model_cfg.get("implementation", ""))
        model = cls.load(checkpoint)
    else:
        typer.echo(f"Instantiating model: {model_id}")
        model = _instantiate_model(model_cfg)

    typer.echo(f"Running predictions on {len(df)} examples…")
    preds = model.predict_batch(df[text_col].tolist())
    out = df.copy()
    out["pred"] = [p.lang_code for p in preds]
    out["confidence"] = [p.confidence for p in preds]
    out["raw_output"] = [p.raw_output for p in preds]

    metrics_dict = _compute_metrics_bundle(out, lang_col)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{model_id}_{dataset_id}_{ts}"
    predictions_path = results_dir / "predictions" / f"{run_id}.parquet"
    metrics_path = results_dir / "metrics" / f"{run_id}.json"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(predictions_path, index=False)
    metrics_path.write_text(json.dumps(metrics_dict, indent=2, default=str))

    run_record = {
        "run_id": run_id,
        "task": "eval",
        "model": model_id,
        "model_version": model_cfg.get("version"),
        "dataset": dataset_id,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "config_hash": _config_hash(model_cfg, exp_cfg),
        "seed": exp_cfg.get("sampling", {}).get("seed"),
        "timestamp": ts,
        "git_commit": _git_commit(),
        "predictions_path": str(predictions_path.relative_to(_REPO_ROOT)),
        "metrics_path": str(metrics_path.relative_to(_REPO_ROOT)),
        "macro_f1": metrics_dict["macro_f1"],
        "n": int(len(out)),
    }
    runs_path = results_dir / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a") as f:
        f.write(json.dumps(run_record, default=str) + "\n")

    typer.echo("")
    typer.echo(f"  macro F1     : {metrics_dict['macro_f1']:.4f}")
    typer.echo(f"  n examples   : {len(out)}")
    typer.echo(f"  n languages  : {out[lang_col].nunique()}")
    typer.echo(f"  predictions  : {predictions_path}")
    typer.echo(f"  metrics      : {metrics_path}")
    typer.echo(f"  run record   : {runs_path}")


def _train_impl(
    model_id: str,
    dataset_id: str,
    config_path: Path | None,
    limit: int | None,
) -> None:
    results_dir = _REPO_ROOT / "results"
    model_cfg = _load_model_cfg(model_id)
    if model_cfg.get("kind") != "trained":
        raise typer.BadParameter(
            f"train requires kind=trained model; got kind={model_cfg.get('kind')!r} for {model_id!r}"
        )

    exp_cfg: dict[str, Any] = {}
    if config_path is not None:
        if not config_path.exists():
            raise typer.BadParameter(f"experiment config not found: {config_path}")
        exp_cfg = _load_yaml(config_path)

    typer.echo(f"Loading dataset: {dataset_id}")
    train_df, eval_df, text_col, lang_col = _resolve_train_dataset(dataset_id, exp_cfg)
    if limit is not None:
        train_df = train_df.head(limit).reset_index(drop=True)
        typer.echo(f"  --limit {limit}: train kept {len(train_df)} rows")
    typer.echo(f"  train: {len(train_df)} rows | eval: {len(eval_df)} rows")

    typer.echo(f"Instantiating model: {model_id}")
    model = _instantiate_model(model_cfg)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{model_id}_{dataset_id}_{ts}"
    model_dir = results_dir / "models" / run_id
    model_dir.mkdir(parents=True, exist_ok=True)

    typer.echo("Fitting model…")
    from language_id.models.trained._sklearn_base import SklearnPipelineLIDModel

    if isinstance(model, SklearnPipelineLIDModel):
        model.fit(train_df, text_col=text_col, lang_col=lang_col)
        artifact_path = model_dir / "model.joblib"
        model.save(artifact_path)
    else:
        # Assume XLMRModel signature.
        model.fit(
            train_df=train_df,
            eval_df=eval_df,
            text_col=text_col,
            lang_col=lang_col,
            output_dir=model_dir,
        )
        artifact_path = model_dir
        model.save(model_dir)

    typer.echo(f"Evaluating on dev ({len(eval_df)} rows)…")
    preds = model.predict_batch(eval_df[text_col].tolist())
    out = eval_df.copy()
    out["pred"] = [p.lang_code for p in preds]
    out["confidence"] = [p.confidence for p in preds]
    out["raw_output"] = [p.raw_output for p in preds]

    metrics_dict = _compute_metrics_bundle(out, lang_col)
    metrics_path = model_dir / "dev_metrics.json"
    metrics_path.write_text(json.dumps(metrics_dict, indent=2, default=str))

    run_record = {
        "run_id": run_id,
        "task": "train",
        "model": model_id,
        "model_version": model_cfg.get("version"),
        "dataset": dataset_id,
        "config_hash": _config_hash(model_cfg, exp_cfg),
        "timestamp": ts,
        "git_commit": _git_commit(),
        "artifact_path": str(artifact_path.relative_to(_REPO_ROOT)),
        "metrics_path": str(metrics_path.relative_to(_REPO_ROOT)),
        "dev_macro_f1": metrics_dict["macro_f1"],
        "n_train": int(len(train_df)),
        "n_dev": int(len(eval_df)),
    }
    runs_path = results_dir / "runs.jsonl"
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with runs_path.open("a") as f:
        f.write(json.dumps(run_record, default=str) + "\n")

    typer.echo("")
    typer.echo(f"  dev macro F1 : {metrics_dict['macro_f1']:.4f}")
    typer.echo(f"  artifact     : {artifact_path}")
    typer.echo(f"  dev metrics  : {metrics_path}")
    typer.echo(f"  run record   : {runs_path}")
    typer.echo("")
    typer.echo(
        f"Evaluate on another dataset with:\n"
        f"  language-id eval --model {model_id} --dataset <id> --checkpoint {artifact_path}"
    )


def _add_language_impl(dataset_id: str, config_path: Path | None) -> None:
    from language_id.experiments.exp3_byodataset import (
        _plot_curve,
        _run_data_efficiency,
        _split_dataset,
    )

    cfg_path = config_path or (
        _REPO_ROOT / "configs" / "experiments" / "exp3_byodataset.yaml"
    )
    cfg = _load_yaml(cfg_path)
    de_cfg = cfg.get("data_efficiency")
    if not de_cfg:
        raise typer.BadParameter(
            f"config {cfg_path} has no `data_efficiency:` block"
        )

    df = load_dataset_by_id(dataset_id)
    if df.empty:
        raise typer.BadParameter(f"no rows loaded from dataset {dataset_id!r}")
    text_col = resolve_text_col(df)
    if df["lang"].nunique() == 1:
        typer.echo(
            f"[add-language] WARNING: dataset is monolingual ({df['lang'].iloc[0]}); "
            "the resulting curve only measures fit-to-one-class behavior. "
            "Mix in background languages for a meaningful classifier."
        )

    train_pool, dev_df, test_df = _split_dataset(df)
    typer.echo(
        f"[add-language] dataset={dataset_id} train={len(train_pool)} "
        f"dev={len(dev_df)} test={len(test_df)} (text_col={text_col!r})"
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_id = dataset_id.replace("/", "_")
    out_root = _REPO_ROOT / "results" / "add_language" / f"{safe_id}_{ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    de_df = _run_data_efficiency(
        de_cfg, train_pool, dev_df, test_df, text_col=text_col, lang_col="lang"
    )
    parquet_path = out_root / "data_efficiency.parquet"
    de_df.to_parquet(parquet_path, index=False)
    plot_path = out_root / "data_efficiency.png"
    _plot_curve(de_df, plot_path)

    summary = {
        "dataset": dataset_id,
        "n_total": int(len(df)),
        "n_train": int(len(train_pool)),
        "n_dev": int(len(dev_df)),
        "n_test": int(len(test_df)),
        "languages_present": sorted(df["lang"].unique().tolist()),
        "config": str(cfg_path.relative_to(_REPO_ROOT)) if cfg_path.is_relative_to(_REPO_ROOT) else str(cfg_path),
        "timestamp": ts,
        "git_commit": _git_commit(),
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    typer.echo("")
    typer.echo(f"  parquet : {parquet_path}")
    typer.echo(f"  plot    : {plot_path}")
    typer.echo(f"  summary : {out_root / 'summary.json'}")


if __name__ == "__main__":
    app()
