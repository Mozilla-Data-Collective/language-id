# Experiment 3 — Out-of-distribution case study (Marma) + data-efficiency curve

Marma appears in neither benchmark. Two parts:

- **Zero-shot and few-shot LLM evaluation** on Marma test data.
- **Data-efficiency curve**: train each of the three classifiers from Experiment 2 on `n ∈ {10, 50, 100, 500, 1000}` Marma examples, plot accuracy vs. `n`. This is the empirical backing for [Experiment 4](4-add-your-language.md).

## Running

```bash
uv run python -m language_id.experiments.exp3_byodataset configs/experiments/exp3_byodataset.yaml
```

The experiment loads whatever dataset ID is set under `dataset:` in the config (via `datacollective`, exactly like CommonLID / CommonVoiceLID). Marma is just the default; swap in any Mozilla Data Collective dataset ID — that's also how [Experiment 4](4-add-your-language.md) brings in a new language.

!!! note
    Set the Marma `dataset:` ID in `configs/experiments/exp3_byodataset.yaml` before the first run.
