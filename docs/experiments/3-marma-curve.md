# Experiment 3 — Out-of-distribution case study (Marma) + data-efficiency curve

Marma appears in neither benchmark. Two parts:

- **Zero-shot and few-shot LLM evaluation** on Marma test data.
- **Data-efficiency curve**: train each of the three classifiers from Experiment 2 on `n ∈ {10, 50, 100, 500, 1000}` Marma examples, plot accuracy vs. `n`. This is the empirical backing for [Experiment 4](4-add-your-language.md).

## Running

```bash
uv run python -m language_id.experiments.exp3_marma_curve configs/experiments/exp3_marma_curve.yaml
```

!!! note
    The Marma loader name in `datacollective` is an open item (spec §16). The first run will surface the exact loader identifier; pin it in `configs/experiments/exp3_marma_curve.yaml`.
