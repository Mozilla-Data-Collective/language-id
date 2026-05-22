# language-id

A reproducible benchmarking project comparing frontier LLMs, classical tools, and custom-trained classifiers on text-based language identification across the web and transcribed-speech domains.

Built around two open benchmarks — [CommonLID](https://commoncrawl.org/) (Common Crawl Foundation) and CommonVoiceLID (Mozilla Data Collective) — with explicit support for adding new languages.

## Why

Text Language Identification (LID) is foundational to multilingual NLP — every downstream pipeline that touches multilingual data (training corpora, evaluation, accessibility tooling, search) depends on it. This project quantifies the high-resource / low-resource performance gap and provides tooling to extend coverage.

## Experiments

1. [Off-the-shelf evaluation on CommonLID](experiments/1-offshelf-eval.md)
2. [Train classifiers on CommonVoiceLID](experiments/2-train-classifiers.md)
3. [Out-of-distribution case study (Marma) + data-efficiency curve](experiments/3-marma-curve.md)
4. [Add your own language](experiments/4-add-your-language.md)
5. [Cross-benchmark evaluation](experiments/5-cross-benchmark.md)

See [Results](results.md) for the latest benchmark numbers, and [Architecture](architecture.md) for the system design.

## Quickstart

```bash
uv sync
# Configure API keys for the LLM providers you want to evaluate:
echo "MISTRAL_API_KEY=..." >> .env
echo "TOGETHER_API_KEY=..." >> .env   # for Gemma / Qwen via Together
# datacollective for dataset access:
echo "DATACOLLECTIVE_API_KEY=..." >> .env
```

Run the CLI:

```bash
# Evaluate a classical baseline end-to-end:
uv run language-id eval --model langdetect --dataset commonvoice_lid --limit 500

# Train a classifier (writes results/models/<run_id>/):
uv run language-id train --model logreg --dataset commonvoice_lid --limit 100000

# Cross-benchmark eval (uses the latest train run):
uv run language-id eval --model logreg --dataset commonlid \
    --checkpoint results/models/<latest-run-id>

# Run an experiment orchestrator:
uv run python -m language_id.experiments.exp3_marma_curve \
    configs/experiments/exp3_marma_curve.yaml

# Regenerate figures + docs/results.md from results/:
uv run language-id report
```

## CLI overview

| Command | Purpose |
| --- | --- |
| `eval` | Run a single (model, dataset) eval; writes parquet + metrics JSON. |
| `train` | Fit a trained classifier; saves to `results/models/`. |
| `add-language` | Run the [add-a-language](experiments/4-add-your-language.md) data-efficiency loop on a user-supplied corpus. |
| `report` | Regenerate `docs/figures/` and `docs/results.md` from `results/`. |
| `compute-tiers` | Recompute per-language resource tiers (`src/language_id/languages/_tiers.json`). |
| `cache (clear\|stats\|inspect)` | Manage the disk-backed LLM cache. |
