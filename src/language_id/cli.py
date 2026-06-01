"""
CLI: evaluate a model on a sampled dataset, per language.

    language-id eval --model qwen --n 10
    language-id eval --model langdetect --dataset commonlid --n 50 --langs eng,kab
    language-id models
"""

from pathlib import Path

import typer

from language_id.data import (
    load_commonlid,
    load_commonvoice_lid,
    load_dataset_by_id,
    sample,
)
from language_id.evaluate import evaluate
from language_id.models import available_models, get_model
from language_id.reporting import save_run

app = typer.Typer(
    name="language-id",
    help="Benchmark text language identification.",
    no_args_is_help=True,
    add_completion=False,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TABLE_NAME = "lang_codes_mapping.csv"


def _load(dataset: str):
    if dataset == "commonlid":
        return load_commonlid()
    if dataset == "commonvoice_lid":
        return load_commonvoice_lid(split="test")
    return load_dataset_by_id(dataset)


@app.command()
def eval(
    model: str = typer.Option(..., "--model", "-m", help="Model name (see `language-id models`)."),
    n: int = typer.Option(10, "--n", help="Samples per language."),
    dataset: str = typer.Option(
        "commonlid", "--dataset", help="'commonlid', 'commonvoice_lid', or a datacollective ID."
    ),
    langs: str | None = typer.Option(
        None, "--langs", help="Comma-separated languages to restrict to (codes or names)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save predictions, metrics, and graphs to results/."
    ),
) -> None:
    """Sample `n` rows per language and evaluate `model` on them."""
    typer.echo(f"Loading dataset: {dataset}")
    df = _load(dataset)
    lang_list = [s.strip() for s in langs.split(",")] if langs else None
    df = sample(df, n_per_lang=n, langs=lang_list, seed=seed)
    typer.echo(f"Sampled {len(df)} rows across {df['lang'].nunique()} languages.")

    typer.echo(f"Evaluating model: {model}")
    overall, per_lang, predictions = evaluate(df, get_model(model))

    typer.echo("")
    typer.echo(f"  accuracy  : {overall['accuracy']:.4f}")
    typer.echo(f"  macro F1  : {overall['macro_f1']:.4f}")
    typer.echo(f"  n         : {overall['n']}  ({overall['n_languages']} languages)")
    typer.echo("")
    typer.echo(per_lang.sort_values("f1", ascending=False).to_string(index=False))

    if save:
        out_dir = _REPO_ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = save_run(out_dir, model, dataset, overall, per_lang, predictions)
        typer.echo(f"\nSaved predictions, metrics, and graphs to {run_dir}")


@app.command()
def models() -> None:
    """List available model names."""
    for name in available_models():
        typer.echo(name)


def _fmt(codes: set[str]) -> str:
    from language_id.lang_codes_mapping import language_name

    rows = sorted((language_name(c), c) for c in codes)
    return "\n".join(f"  {c:<8} {name}" for name, c in rows) or "  (none)"



if __name__ == "__main__":
    app()
