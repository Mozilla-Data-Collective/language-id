from pathlib import Path

import typer

from language_id.data import (
    load_commonlid,
    load_commonvoice_lid,
    load_dataset_by_id,
    load_single_language_dataset,
    sample,
)
from language_id.evaluate import evaluate
from language_id.models import available_models, get_model
from language_id.reporting import save_run

app = typer.Typer(
    name="language-id",
    help="Benchmark and train text language identification models on Mozilla Data Collective datasets.",
    no_args_is_help=True,
    add_completion=False,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(
    dataset: str,
    lang_col: str = "tag",
    text_col: str = "sentence",
    ground_truth_language: str | None = None,
):
    if dataset == "commonlid":
        return load_commonlid()
    if dataset == "commonvoice_lid":
        return load_commonvoice_lid(split="test")
    if ground_truth_language is not None:
        return load_single_language_dataset(dataset, ground_truth_language, text_col_name=text_col)
    return load_dataset_by_id(dataset, lang_col_name=lang_col, text_col_name=text_col)


@app.command()
def eval(
    model: str = typer.Option(..., "--model", "-m", help="Model name (see `language-id models`)."),
    n: int = typer.Option(0, "--n", help="Samples per language. Use 0 for the whole dataset."),
    dataset: str = typer.Option(
        "commonlid", "--dataset", help="'commonlid', 'commonvoice_lid', "
                                       "or a Mozilla Data Collective dataset ID."
    ),
    lang_col: str = typer.Option(
        "tag",
        "--lang-col",
        help="Language column name in a custom dataset ID.",
    ),
    text_col: str = typer.Option(
        "sentence",
        "--text-col",
        help="Text column name in a custom dataset ID.",
    ),
    ground_truth_language: str | None = typer.Option(
        None,
        "--ground-truth-language",
        help="Treat the dataset as single-language: every row's gold label is this "
        "language (any code/name form). The dataset only needs a text column and --lang-col is ignored.",
    ),
    langs: str | None = typer.Option(
        None, "--langs", help="Comma-separated languages to restrict to (codes or names)."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling seed."),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Save predictions, metrics, and graphs to results/."
    ),
) -> None:
    """Evaluate `model` on the dataset. By default, samples `n` rows per language; `--n 0` uses every row."""
    typer.echo(f"Loading dataset: {dataset}")
    df = _load(dataset, lang_col=lang_col, text_col=text_col, ground_truth_language=ground_truth_language)
    lang_list = [s.strip() for s in langs.split(",")] if langs else None
    if n > 0:
        df = sample(df, n_per_lang=n, langs=lang_list, seed=seed)
        typer.echo(f"Sampled {len(df)} rows across {df['lang'].nunique()} languages.")
    else:
        if lang_list is not None:
            df = sample(df, n_per_lang=len(df), langs=lang_list, seed=seed)
        typer.echo(f"Using whole dataset: {len(df)} rows across {df['lang'].nunique()} languages.")

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



if __name__ == "__main__":
    app()
