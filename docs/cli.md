# CLI reference

The package installs a single command, `language-id`, with three subcommands:

```bash
uv run language-id --help
```

| Command | Purpose |
|---|---|
| [`eval`](#language-id-eval) | Evaluate an LID model on a dataset and save per-language metrics, predictions, and plots. |
| [`train`](#language-id-train) | Train a single-language detector from a text corpus and score it on a held-out test split. |
| [`eval-models`](#language-id-eval-models) | List the eval model names accepted by `eval --eval-model`. |

!!! note "Eval models vs. train models"
    The CLI distinguishes two kinds of "models":

    - **Eval models** (`eval --eval-model`) — ready-made, off-the-shelf models you *evaluate*: the standard tools (langdetect, GlotLID, NLLB-LID) and LLMs. List them with `language-id eval-models`.
    - **Train models** (`train --train-model`) — the kinds of models you *train* yourself: `naive_bayes`, `logreg`, or a fine-tuned Hugging Face `llm`.

    See [Models](models.md) for details on both families.

## `language-id eval`

Evaluate a model on a dataset. By default it samples `--n` rows per language; `--n 0` uses every row.

```bash
# Standard tool on the CommonLID benchmark, 200 rows per language
uv run language-id eval --eval-model glotlid --dataset commonlid --n 200

# LLM with 3 few-shot examples, restricted to two languages
uv run language-id eval --eval-model gpt-oss-120b --shot 3 --langs "lad,nah" --n 100

# Any MDC dataset by ID, specifying its column names
uv run language-id eval --eval-model nllb-lid --dataset your-dataset-id --lang-col tag --text-col sentence

# A single-language corpus: every row's gold label is the given language
uv run language-id eval --eval-model glotlid --dataset your-dataset-id --ground-truth-language lad
```

### Options

| Option | Default     | Description                                                                                                                                                         |
|---|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--eval-model`, `-m` | (required)  | Eval model name, see `language-id eval-models` and [Models](models.md).                                                                                                       |
| `--n` | `0`         | Samples per language. `0` evaluates on the whole dataset.                                                                                                           |
| `--shot` | `0`         | Few-shot demonstrations (0–10) prepended to the LLM prompt, only applicable to LLMs. The examples are held out from the evaluation set so they never leak into it.  |
| `--dataset` | `commonlid` | `commonlid`, `commonvoice_lid`, or any MDC dataset ID of a text corpus, see [Datasets](datasets.md).                                                                |
| `--lang-col` | `tag`       | Language column name when using a custom dataset ID.                                                                                                                |
| `--text-col` | `sentence`  | Text column name when using a custom dataset ID.                                                                                                                    |
| `--ground-truth-language` | -           | Treat the dataset as single-language: every row's gold label is this language (any code or name form), so only a text column is needed and `--lang-col` is ignored. |
| `--langs` | -           | Comma-separated languages (codes or names) to restrict the experiments if the dataset contains multiple languages.                                                 |
| `--seed` | `0`         | Sampling seed.                                                                                                                                                      |
| `--save/--no-save` | `--save`    | Persist predictions, metrics, and graphs to `results/`.                                                                                                             |

### Output

Each run prints overall accuracy and macro F1 plus a per-language table, and (with `--save`) writes a timestamped directory:

```
results/<model>_<dataset>_<timestamp>/
    predictions.csv               every row with gold, pred, confidence, raw output
    per_language.csv              per-language support / recall / precision / f1
    metrics.json                  overall + per-language metrics
    plots/per_language_f1.png
    plots/per_language_metrics.png
    plots/confusion_matrix.png
```

A one-line summary is also appended to `results/runs.jsonl`, which the [compare-saved-runs notebook](notebooks.md) uses to compare runs across models.

## `language-id train`

Train a detector for a single target language and report precision/recall/F1 on a held-out test split. 
The command turns a single-language text corpus into a binary problem: *is this sentence the target language or not?* 
by pairing its sentences (positives) with other-language sentences from Common Voice LID (negatives).

```bash
# Train a character n-gram Naive Bayes model on a custom dataset with a custom target language
uv run language-id train --dataset your-dataset-id --train-model naive_bayes --lang your-language-code-or-name

# Fine-tune a Hugging Face model instead (needs the `finetune` extra)
uv run language-id train --dataset your-dataset-id --lang lad \
    --train-model llm --hf-model-id Qwen/Qwen3-0.6B --epochs 3 --batch-size 8
```

### Options

| Option | Default           | Description                                                                                               |
|---|-------------------|-----------------------------------------------------------------------------------------------------------|
| `--dataset` | (required)        | Single-language dataset ID/slug on MDC.                                                                   |
| `--lang` | (required)        | Target language of every row, in any code or name form (e.g. `lad`).                                      |
| `--train-model` | `naive_bayes`     | `naive_bayes`, `logreg` (char n-gram naive bayes or logistic regression) or `llm` (fine-tune a HF model). |
| `--hf-model-id` | `Qwen/Qwen3-0.6B` | Hugging Face model ID to fine-tune (`--train-model llm` only).                                                 |
| `--epochs` | `3`               | Fine-tuning epochs (LLM only).                                                                            |
| `--batch-size` | `8`               | Fine-tuning batch size (LLM only).                                                                        |
| `--n-train` | `0`               | Number of target-language training samples. `0` uses every positive row.                                |
| `--n-neg` | `0`               | Number of Common Voice LID negatives. `0` matches the number of positives 1:1.                            |
| `--seed` | `0`               | Sampling/training seed.                                                                                   |

## `language-id eval-models`

Lists every eval model name accepted by `eval --eval-model`, i.e. the standard tools plus the LLMs currently registered. See [Models](models.md) for what each one is.

```bash
uv run language-id eval-models
```
