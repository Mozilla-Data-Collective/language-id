# Architecture

This page describes the system design at a high level. The authoritative source is [`spec.md`](https://github.com/Mozilla-Data-Collective/language-id/blob/main/spec.md) in the repo root.

## The `LIDModel` protocol

Everything — frontier LLMs, langdetect, trained XLM-R — implements one interface (`src/language_id/models/base.py`):

```python
@dataclass
class LIDPrediction:
    lang_code: str          # BCP-47, e.g. "en", "zh-Hans", "kab"
    confidence: float | None # 0.0-1.0 if available, else None
    raw_output: str          # for audit, especially for LLMs

class LIDModel(Protocol):
    name: str
    version: str
    def predict(self, text: str) -> LIDPrediction: ...
    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]: ...
```

The evaluation loop is identical regardless of model kind. Trained models additionally expose `fit()`, `save()`, `load()`, and `predict_proba()` where applicable.

## Length-stratified sampling

The primary motivation is apples-to-apples comparison between CommonLID and CommonVoiceLID at matched sentence-length distributions per language. Bucket boundaries are defined in WORDS, and reporting uses the *same* bucket definitions as sampling (single source of truth).

For non-spaced scripts (zh*, ja, th, km, lo, my, bo), word-counting via whitespace gives nonsense. The `count_words(text, lang_code)` function in `data/length_buckets.py` applies a per-language character-to-word scaling factor from `configs/word_count_overrides.yaml`.

## Language code normalization

All loaders normalize language codes to BCP-47 at load time (Common Voice standard). A versioned mapping table in `src/language_id/languages/_mapping_overrides.yaml` handles known mismatches between datasets and model outputs. The full reconciliation table is rendered into [Language code reconciliation](language-code-reconciliation.md).

## Reproducibility

Every run records: model snapshot ID, prompt template hash, config hash, random seeds, dataset version, git commit, and W&B URL. Stored in `results/runs.jsonl` and W&B run config.

## Module layout

| Module | Responsibility |
| --- | --- |
| `data/` | Dataset loaders, BCP-47 normalization, length-stratified sampling. |
| `languages/` | BCP-47 helpers, ISO 15924 script families, resource-tier lookup. |
| `models/base.py` | The `LIDModel` protocol + `LIDPrediction` dataclass. |
| `models/classical/` | langdetect, GlotLID, NLLB-LID (fastText-based). |
| `models/llm/` | `AnyLLMModel` wrapping `any-llm`; shared retry/cache/parse base. |
| `models/trained/` | LogReg, NGramNB (sklearn pipeline) and XLM-R LoRA (HF Trainer + PEFT). |
| `parsing/llm_output.py` | Strip prefixes → `langcodes.find` → alias table → unparseable. |
| `caching/llm_cache.py` | `diskcache` keyed by `(model, version, prompt_hash, text_hash)`. |
| `metrics/` | Macro-F1, per-language F1, sliced metrics (length/script/tier), calibration. |
| `reporting/` | `figures.py` and `tables.py` — regenerate `docs/figures/` + `docs/results.md`. |
| `experiments/` | Thin orchestrators for Exp 1, 2, 3, 5 (Exp 4 is the `add-language` CLI). |
| `cli.py` | Typer entry point; `eval`, `train`, `add-language`, `report`, `cache`, `compute-tiers`. |

## End-to-end flow

```
   datacollective.load_dataset
            │
            ▼
   data/loaders.py  ──→  normalize_lang_column  ──→  pandas.DataFrame
            │
            ▼  (optional) length_stratified_sample
            │
            ▼
   model.predict_batch (LIDModel protocol)
            │
            ▼
   metrics/core + metrics/sliced
            │
            ├─→ results/predictions/<run_id>.parquet
            ├─→ results/metrics/<run_id>.json
            └─→ results/runs.jsonl  ─→  reporting/  ─→  docs/figures/ + docs/results.md
```

## Open spec items

The following are flagged in `spec.md` §16; resolve before publishing benchmark numbers:

- **NLLB-200 LID weights** — confirm the canonical HF repo & filename.
- **`fasttext` dep** — required by GlotLID and NLLB-LID. Not in the default deps; add `fasttext` (or `fasttext-wheel`) to enable them.
- **LLM snapshot IDs** — `configs/models/{gpt-5,claude-opus-4,qwen-3,mistral-large}.yaml` carry `version: TBD` and (for Qwen) `provider: TBD`. Lock at experiment start.
- **Word-count overrides** for non-spaced scripts — defaults provided; tune if empirical results look off.
