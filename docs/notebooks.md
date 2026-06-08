# Notebooks

The notebooks under [`src/language_id/notebooks/`](https://github.com/Mozilla-Data-Collective/language-id/tree/main/src/language_id/notebooks) are interactive walkthroughs of the same building blocks the CLI uses. Launch them with your editor of choice or with Jupyter:

```bash
uv run jupyter lab src/language_id/notebooks/
```

All notebooks read the API keys from the repo-root `.env` file (see [Getting started](getting-started.md)).

## Run model evaluation

**`run-model-evaluation.ipynb`** — benchmark several LID models side by side in one go.

- Pick a dataset (`commonlid` or `commonvoice_lid`), a per-language sample size, and a set of models: standard tools (langdetect, GlotLID, NLLB-LID) and/or LLMs served via Together.
- LLMs can be run at several few-shot settings (e.g. 0-shot vs 1-shot) in the same sweep; the few-shot examples are held out from the evaluation set.
- Produces an overview comparison table and figure, a per-language F1 heatmap, and a "where models disagree most" breakdown.
- Each individual evaluation is saved to `results/` in the same format as `language-id eval`, so runs can be revisited later.

## Compare saved runs

**`compare-saved-runs.ipynb`** — post-hoc analysis of runs you have already computed.

- Every `language-id eval` run (and every run from the evaluation notebook) is saved under `results/<model>_<dataset>_<timestamp>/`. This notebook discovers those runs, lets you pick a set of them, and compares them side by side **without re-running any model**.
- Answers questions like: *given the same dataset and setup, which model was best overall, and where does each one win or lose per language?*
- Includes a comparability check: only runs on the same dataset with the same per-language support are compared, so the comparison stays fair.
- Outputs an overview table, a per-language heatmap, and a disagreement plot; comparison figures are written to `results/comparisons/`.

## Train a single-language detector

**`train-single-language-detector.ipynb`** — build a specialist detector for one (typically low-resource) language and see how it stacks up against off-the-shelf models.

- Starts from a single-language text corpus on MDC (the notebook uses Ladino as the running example) and pairs it with Common Voice LID negatives to form a binary detection problem.
- Trains a fast CPU classifier (character n-gram naive Bayes), and optionally fine-tunes a Hugging Face model (e.g. `Qwen/Qwen3-0.6B`; needs the `finetune` extra) as a stronger specialist.
- Evaluates every detector — including baselines such as GlotLID and zero-shot LLMs — on the same held-out test split and plots their detection scores side by side.
- The same workflow is available non-interactively as `language-id train` (see the [CLI reference](cli.md)).

## Train and evaluate a local LID model

**`train-and-evaluate-local-lid.ipynb`** — a tutorial that builds a multi-class language detector locally and explains the process step-by-step.

- Trains a single character n-gram + Multinomial Naive Bayes classifier over **all** languages in Common Voice LID (300+).
- Walks through the modeling choices: why character n-grams beat word features for LID, why a `HashingVectorizer` keeps memory constant on a 20M-row corpus, and how Naive Bayes (with Laplace smoothing and optional `partial_fit` online learning) fits the problem. An `SGDClassifier` linear-SVM alternative is included as a commented swap-in.
- Downloads and subsamples CV-LID (default 15%) into train/dev/test splits by streaming the TSV in chunks, then evaluates on both the held-out CV-LID dev split and the cross-domain CommonLID benchmark, including a plot of how differently the two datasets distribute languages.
- Includes error-analysis helpers like `show_worst_languages` ranks languages by F1 (with a minimum-support filter to ignore tiny test sets) and `show_confusions` shows which languages a given one is mistaken for, to make data-scarcity and closely-related-language failures visible.
- A good starting point if you want to understand the mechanics behind `language-id train` before using the CLI.

## Add a new language to LID

**`add-new-language-to-lid.ipynb`** — bootstrap support for a language that an off-the-shelf model handles poorly (or not at all) by folding your own corpus into the training and evaluation data.

- Builds on the local training pipeline above: it sets a `TARGET_LANG` (ISO 639-3), checks whether that language is already present in CV-LID, and reports whether you are reinforcing an existing language or adding one from scratch.
- Loads a single-language corpus from MDC (the notebook uses Ladino, with a Gujarati variant shown in commented cells), concatenates it into the CV-LID training set, and appends a held-out test set for the language to CommonLID.
- Retrains the same character n-gram Naive Bayes classifier on the augmented data and prints a `classification_report` so you can see the new language's precision/recall/F1 alongside the existing ones.
- Useful for quickly answering *"how much does a modest amount of in-language text move the needle?"* before committing to a full single-language detector or fine-tune.
