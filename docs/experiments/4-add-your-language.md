# Experiment 4 — Add your own language

Not a separate experiment scientifically — it's tooling reuse on top of [Experiment 3](3-marma-curve.md).

The `language-id add-language` CLI wraps the data-efficiency workflow so you can drop in a labeled corpus for a new language and immediately get a trained classifier plus a `n` vs. accuracy curve.

## Running

```bash
uv run language-id add-language --code <bcp47> --data <path-to-your-corpus>
```

## What you need

- A labeled corpus for the target language (sentences + BCP-47 language tag).
- An obvious BCP-47 code for the language.

## What you get

- A trained classifier saved to `results/models/`.
- A data-efficiency plot under `docs/figures/`.
- A line in `results/runs.jsonl` with the seed and config hash for reproducibility.
