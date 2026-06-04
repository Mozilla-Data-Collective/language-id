from pathlib import Path

import typer

from language_id.data import (
    load_commonlid,
    load_commonvoice_lid,
    load_dataset_by_id,
    load_single_language_dataset,
    sample,
    take_fewshot_examples,
)
from language_id.eval_models import TOGETHER_MODELS, available_eval_models, get_eval_model
from language_id.evaluate import evaluate
from language_id.reporting import save_run
from language_id.train import (
    DEFAULT_HF_MODEL_ID,
    build_training_data,
    evaluate_detector,
    finetune_llm,
    train_logreg,
    train_naive_bayes,
)

MAX_SHOTS = 10

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
    eval_model: str = typer.Option(
        ...,
        "--eval-model",
        "-m",
        help="Eval model: an off-the-shelf model to evaluate (see `language-id eval-models`). "
        "Not to be confused with the train models of `language-id train`.",
    ),
    n: int = typer.Option(0, "--n", help="Samples per language. Use 0 for the whole dataset."),
    shot: int = typer.Option(
        0,
        "--shot",
        min=0,
        max=MAX_SHOTS,
        help=f"Few-shot demonstrations to prepend to the LLM prompt (0-{MAX_SHOTS}). "
        "0 is zero-shot. Only applies to LLMs; examples are held out from evaluation.",
    ),
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
    """Evaluate `eval_model` on the dataset. By default, samples `n` rows per language; `--n 0` uses every row."""
    if shot > 0 and eval_model not in TOGETHER_MODELS:
        raise typer.BadParameter(f"--shot is only supported for LLMs, not {eval_model!r}.")

    typer.echo(f"Loading dataset: {dataset}")
    df = _load(dataset, lang_col=lang_col, text_col=text_col, ground_truth_language=ground_truth_language)

    # Hold out the few-shot demonstrations before sampling so they never leak
    # into the evaluation set.
    examples, df = take_fewshot_examples(df, shot, seed=seed)
    if examples:
        shown = ", ".join(code for _, code in examples)
        typer.echo(f"Few-shot: {shot}-shot (example languages: {shown})")

    lang_list = [s.strip() for s in langs.split(",")] if langs else None
    if n > 0:
        df = sample(df, n_per_lang=n, langs=lang_list, seed=seed)
        typer.echo(f"Sampled {len(df)} rows across {df['lang'].nunique()} languages.")
    else:
        if lang_list is not None:
            df = sample(df, n_per_lang=len(df), langs=lang_list, seed=seed)
        typer.echo(f"Using whole dataset: {len(df)} rows across {df['lang'].nunique()} languages.")

    # Label few-shot runs distinctly so saved results compare cleanly (e.g. 0- vs 2-shot).
    label = f"{eval_model}-{shot}shot" if shot else eval_model
    model_obj = get_eval_model(eval_model, examples=examples or None)
    model_obj.name = label
    typer.echo(f"Evaluating eval model: {label}")
    overall, per_lang, predictions = evaluate(df, model_obj)

    typer.echo("")
    typer.echo(f"  accuracy  : {overall['accuracy']:.4f}")
    typer.echo(f"  macro F1  : {overall['macro_f1']:.4f}")
    typer.echo(f"  n         : {overall['n']}  ({overall['n_languages']} languages)")
    typer.echo("")
    typer.echo(per_lang.sort_values("f1", ascending=False).to_string(index=False))

    if save:
        out_dir = _REPO_ROOT / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_dir = save_run(out_dir, label, dataset, overall, per_lang, predictions)
        typer.echo(f"\nSaved predictions, metrics, and graphs to {run_dir}")


@app.command()
def train(
    dataset: str = typer.Option(
        ..., "--dataset", help="Single-language dataset ID/slug (a .tar.gz text corpus)."
    ),
    lang: str = typer.Option(
        ..., "--lang", help="Target language of every row (any code/name form, e.g. 'lad')."
    ),
    train_model: str = typer.Option(
        "naive_bayes",
        "--train-model",
        help="Train model: the kind of model to train — 'naive_bayes' (char n-gram naive bayes), "
        "'logreg' (char n-gram logistic regression) or 'llm' (fine-tune a HF model). "
        "Not to be confused with the eval models of `language-id eval`.",
    ),
    hf_model_id: str = typer.Option(
        DEFAULT_HF_MODEL_ID,
        "--hf-model-id",
        help="Hugging Face model id to fine-tune (only used with --train-model llm).",
    ),
    epochs: float = typer.Option(3, "--epochs", help="Fine-tuning epochs (LLM only)."),
    batch_size: int = typer.Option(8, "--batch-size", help="Fine-tuning batch size (LLM only)."),
    n_train: int = typer.Option(
        0, "--n-train", help="Cap on target-language training samples. 0 uses every positive row."
    ),
    n_neg: int = typer.Option(
        0, "--n-neg", help="Common Voice LID training negatives. 0 matches the number of positives."
    ),
    seed: int = typer.Option(0, "--seed", help="Sampling/training seed."),
) -> None:
    """Train a single-language detector and report its detection scores on a held-out test split.

    Builds a binary problem (target language vs. Common Voice LID negatives) and fits either a
    char n-gram naive Bayes (`--train-model naive_bayes`), a char n-gram logistic regression
    (`--train-model logreg`) or a fine-tuned HF model
    (`--train-model llm --hf-model-id ...`, needs the `finetune` extra).
    """
    if train_model not in {"naive_bayes", "logreg", "llm"}:
        raise typer.BadParameter("--train-model must be 'naive_bayes', 'logreg' or 'llm'.")

    typer.echo(f"Building training data from {dataset} (target language: {lang}) ...")
    train_df, test_df = build_training_data(
        dataset, lang, n_train=n_train or None, n_neg=n_neg or None, seed=seed
    )
    n_pos_train = int((train_df["label"] != "other").sum())
    typer.echo(
        f"  train: {len(train_df)} rows ({n_pos_train} {lang} / {len(train_df) - n_pos_train} other)"
        f"  |  test: {len(test_df)} rows"
    )

    if train_model == "naive_bayes":
        typer.echo("Training char n-gram Naive Bayes model ...")
        model = train_naive_bayes(train_df)
    elif train_model == "logreg":
        typer.echo("Training char n-gram logistic regression ...")
        model = train_logreg(train_df)
    else:
        typer.echo(f"Fine-tuning Hugging Face model: {hf_model_id} ...")
        model = finetune_llm(
            train_df, lang, model_id=hf_model_id, epochs=epochs, batch_size=batch_size, seed=seed
        )

    scores = evaluate_detector(model, test_df, lang)
    typer.echo("")
    typer.echo(f"  model     : {scores['model']}")
    typer.echo(f"  precision : {scores['precision']:.4f}")
    typer.echo(f"  recall    : {scores['recall']:.4f}")
    typer.echo(f"  f1        : {scores['f1']:.4f}")
    typer.echo(f"  accuracy  : {scores['accuracy']:.4f}")
    typer.echo(f"  n (test)  : {scores['n']}")


@app.command(name="eval-models")
def eval_models() -> None:
    """List available eval model names (used with `eval --eval-model`)."""
    for name in available_eval_models():
        typer.echo(name)



if __name__ == "__main__":
    app()
