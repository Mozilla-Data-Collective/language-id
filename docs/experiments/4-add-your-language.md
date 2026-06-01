# Experiment 4 — Add your own language

Not a separate experiment scientifically — it's tooling reuse on top of [Experiment 3](3-marma-curve.md).

The `language-id add-language` CLI wraps the data-efficiency workflow so you can point at a labeled corpus for a new language and immediately get a trained classifier plus a `n` vs. accuracy curve.

Bringing your own data works exactly like the built-in datasets: upload your corpus to Mozilla Data Collective and pass its **dataset ID** — the same mechanism used for CommonLID, CommonVoiceLID, and Marma.

## Running

```bash
uv run language-id add-language --dataset <mdc-dataset-id>
```

## What you need

- A labeled corpus published as a `datacollective` dataset, with a `lang` column (BCP-47 tags) and a text column (`text` or `sentence`).

## What you get

- A trained classifier saved to `results/models/`.
- A data-efficiency plot under `docs/figures/`.
- A line in `results/runs.jsonl` with the seed and config hash for reproducibility.
