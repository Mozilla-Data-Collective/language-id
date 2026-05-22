# Experiment 5 — Cross-benchmark evaluation

Run the three classifiers from [Experiment 2](2-train-classifiers.md) — trained on CommonVoiceLID — against the full CommonLID test set. Compare to [Experiment 1](1-offshelf-eval.md) baselines.

The framing is the speech → web domain shift: how much of any performance gap is domain mismatch vs. capacity?

## Running

```bash
uv run python -m language_id.experiments.exp5_cross_benchmark configs/experiments/exp5_cross_benchmark.yaml
```
