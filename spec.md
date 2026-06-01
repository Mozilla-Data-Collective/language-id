# LID Benchmark — Implementation Specification

A hands-on, reproducible benchmarking project comparing frontier LLMs, standard tools, and custom-trained classifiers on text-based language identification across the web and transcribed-speech domains. Built around two open benchmarks — **CommonLID** (Common Crawl Foundation) and **CommonVoiceLID** (Mozilla Data Collective) — with explicit support for adding new languages.

This document is the source of truth for implementation. Generate code that matches this spec; if anything here is ambiguous, flag it rather than guessing.

---

## 1. Project context

Text Language Identification (LID) is foundational to multilingual NLP — every downstream pipeline that touches multilingual data (training corpora, evaluation, accessibility tooling, search) depends on it. Significant progress has been made via character n-gram models (langdetect, fastText, CLD3) and more recently neural approaches (XLM-R, GlotLID, NLLB-200 LID), but performance is heavily skewed toward high-resource languages. This project quantifies the gap and provides tooling to extend coverage.

The project is designed to be reproducible, extensible to new languages, and runnable end-to-end by a user.

---

## 2. Datasets

All datasets are accessed through the `datacollective` Python SDK:

```python
from datacollective import load_dataset
df = load_dataset("dataset-id")  # returns pandas.DataFrame
```

### 2.1 CommonLID (evaluation only)
- 109 languages (78 with ≥100 lines)
- ~350,000 lines of human-annotated web text
- Sourced from Common Crawl (CC-MAIN-2024-22, CC-MAIN-2025-05) and MADLAD-400
- **License constraint**: evaluation only; never used for training or hyperparameter tuning

### 2.2 CommonVoiceLID (training-available)
- 19M sentences across 300+ languages
- Built from Mozilla Common Voice scripted (v23) and spontaneous (v1) speech transcriptions
- Pre-split into train/dev/test (use these splits as-is; do not redefine)
- Columns: `id`, `sentence`, `lang`, `sentence_domain`, `source`, `style`, `split`

### 2.3 Marma corpus (OOD case study)
- Accessed via `datacollective` (loader name TBD — implementer should use the obvious identifier and flag if missing)
- Used only in Experiment 3 and as a worked example for Experiment 4
- Right now its the Marma corpus but it should be trivial to change to any other text corpus.



---

## 3. Experiments

### Experiment 1 — Off-the-shelf evaluation on CommonLID
Benchmark these models on a length-stratified subsample of CommonLID (see §7 for sampling spec):

**Frontier LLMs** (lock exact snapshot IDs at experiment start):
- GPT-5.X (OpenAI)
- Claude Opus 4.X (Anthropic)
- Qwen 3.X (32B)
- Mistral ( Mistral Large 2)

**Standard / specialist baselines**:
- `langdetect` (Python package)
- GlotLID
- NLLB-200 LID head

Output normalization to BCP-47 is required. Track unparseable LLM responses as a first-class metric.

LLM inference should be done using the any_llm Python package to enable easy swapping of models and providers. Use the same prompting strategy across all LLMs, and log prompts and responses for reproducibility.
```
pip install 'any-llm-sdk[mistral,ollama]'

export MISTRAL_API_KEY="YOUR_KEY_HERE"  # or OPENAI_API_KEY, etc
from any_llm import completion
import os

# Make sure you have the appropriate environment variable set
assert os.environ.get('MISTRAL_API_KEY')

response = completion(
    model="mistral-small-latest",
    provider="mistral",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```


### Experiment 2 — Train three classifiers on CommonVoiceLID
Three complementary paradigms:
1. **Logistic Regression** on character n-grams (sklearn `Pipeline`: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5))` → `LogisticRegression`)
2. **Multinomial Naive Bayes** on character n-grams (same vectorizer, `MultinomialNB`)
3. **XLM-RoBERTa-large** fine-tuned with **LoRA** (rank 16, target attention projection layers), via HuggingFace `Trainer`

Hyperparameter tuning happens on a held-out slice of CommonVoiceLID `dev`. 

### Experiment 3 — Out-of-distribution case study (Marma) + data-efficiency curve
Marma appears in neither benchmark.

Two parts:
- **Zero-shot and few-shot LLM evaluation** on Marma test data
- **Data-efficiency curve**: train each of the three classifiers from Experiment 2 on `n ∈ {10, 50, 100, 500, 1000}` Marma examples, plot accuracy vs. `n`. This is the empirical backing for Experiment 4.

### Experiment 4 — "Add your own language" pipeline
Documentation + a CLI command (`language-id add-language`) that wraps the Experiment 3 workflow. Not a separate experiment scientifically — it's tooling reuse.

### Experiment 5 — Cross-benchmark evaluation
Run the three classifiers from Experiment 2 against the full CommonLID test set. Compare to Experiment 1 baselines. The framing is the speech→web domain shift: how much of any performance gap is domain mismatch vs. capacity?

---

## 4. Project layout

```
language-id/
├── pyproject.toml
├── README.md
├── mkdocs.yaml
├── .pre-commit-config.yaml
├── .github/workflows/
│   └── docs.yaml                  # build + publish mkdocs to GH Pages
├── docs/
│   ├── index.md
│   ├── architecture.md
│   ├── experiments/
│   │   ├── 1-offshelf-eval.md
│   │   ├── 2-train-classifiers.md
│   │   ├── 3-marma-curve.md
│   │   ├── 4-add-your-language.md
│   │   └── 5-cross-benchmark.md
│   ├── language-code-reconciliation.md
│   ├── results.md
│   └── figures/                   # committed PNG/SVG, regenerated by `report`
├── configs/
│   ├── models/                    # one YAML per model, includes prompts inline
│   ├── experiments/               # one YAML per experiment
│   └── resource_tiers.yaml        # tier thresholds, versioned
├── src/language_id/
│   ├── __init__.py
│   ├── config.py                  # pydantic-settings: API keys, paths, W&B
│   ├── cli.py                     # Typer entry point
│   │
│   ├── data/
│   │   ├── loaders.py             # load_commonlid, load_commonvoice_lid, load_marma
│   │   ├── normalization.py       # all lang codes → BCP-47 (Common Voice standard)
│   │   ├── sampling.py            # length-stratified sampler (§7)
│   │   └── length_buckets.py      # word-count bucketing utilities
│   │
│   ├── languages/
│   │   ├── codes.py               # langcodes wrappers, canonicalization
│   │   ├── script_family.py       # ISO 15924 grouping
│   │   ├── resource_tier.py       # high/mid/low from CommonVoiceLID counts
│   │   └── confusable_pairs.py    # Hindi/Urdu, sr/hr/bs, id/ms, etc.
│   │
│   ├── models/
│   │   ├── base.py                # LIDModel protocol
│   │   ├── llm/
│   │   │   ├── base.py            # shared retry / parse logic
│   │   │   ├── any_llm_client.py
│   │   │   ├── together_client.py
│   │   ├── standard/
│   │   │   ├── langdetect_model.py
│   │   │   ├── glotlid_model.py
│   │   │   └── nllb_lid_model.py
│   │   └── trained/
│   │       ├── logreg.py
│   │       ├── ngram_nb.py
│   │       └── xlmr.py            # LoRA fine-tuning
│   │
│   ├── parsing/
│   │   └── llm_output.py          # free text → BCP-47, tracks unparseable rate
│   │
│   ├── metrics/
│   │   ├── core.py                # macro-F1, per-lang F1, confusion matrix
│   │   ├── sliced.py              # by script family, resource tier, length bucket
│   │   ├── confusable.py          # pair-level accuracy
│   │   └── calibration.py         # ECE, reliability diagrams
│   │
│   ├── tracking/
│   │   └── wandb_logger.py        # consistent run naming, artifacts, sklearn plots
│   │
│   ├── reporting/
│   │   ├── figures.py             # matplotlib/plotly → docs/figures/
│   │   └── tables.py              # parquet → markdown tables
│   │
│   └── experiments/
│       ├── exp1_offshelf_eval.py
│       ├── exp2_train_classifiers.py
│       ├── exp3_byodataset.py
│       └── exp5_cross_benchmark.py
│
└── results/                       # local artifacts (parquet, json) — gitignored
```

---

## 5. Technical decisions (locked)

| Concern               | Decision                                                                                                |
|-----------------------|---------------------------------------------------------------------------------------------------------|
| Package manager       | `uv`                                                                                                    |
| Linter / formatter    | `ruff` (replaces black, isort, flake8)                                                                  |
| Type checker          | `ty` (Astral)                                                                                           |
| Pre-commit            | `ruff` + `ty`                                                                                           |
| Config                | pydantic-settings + YAML (no Hydra)                                                                     |
| CLI                   | Typer, flat style: `language-id eval --model gpt-5 --dataset commonlid`                               |
| Language code library | `langcodes`, canonical form = BCP-47 (Common Voice standard)                                            |
| Tracking              | Weights & Biases (`wandb`, `wandb.sklearn` plots, HF `Trainer` integration via `report_to="wandb"`)     |
| Docs                  | mkdocs + Material theme, static markdown, figures committed under `docs/figures/`                       |
| Tests                 | **None** (explicitly out of scope for now)                                                              |
| Data SDK              | `datacollective`                                                                                        |
| LLM SDK               | `any-llm`                                                                                               |
| XLM-R fine-tuning     | LoRA (rank 16, attention projections). Default; can be overridden to full fine-tune via config.         |
| Resource tier source  | Computed from CommonVoiceLID line counts (own taxonomy, versioned in `configs/resource_tiers.yaml`)     |
| Prompts               | Inline YAML strings in `configs/models/*.yaml`                                                          |
| Reporting             | Static markdown referencing figures committed to `docs/figures/`, regenerated by `language-id report` |

---

## 6. The `LIDModel` protocol

Everything — frontier LLMs, langdetect, trained XLM-R — implements one interface:

```python
from typing import Protocol
from dataclasses import dataclass

@dataclass
class LIDPrediction:
    lang_code: str          # BCP-47, e.g. "en", "zh-Hans", "kab"
    confidence: float | None # 0.0–1.0 if available, else None
    raw_output: str          # for audit, especially for LLMs

class LIDModel(Protocol):
    name: str
    version: str
    def predict(self, text: str) -> LIDPrediction: ...
    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]: ...
```

The evaluation loop is identical regardless of model type. Trained models additionally expose `fit()`, `save()`, `load()`, and `predict_proba()` where applicable.

---

## 7. Length-stratified sampling (parametrizable)

**Primary motivation**: enable apples-to-apples comparison between CommonLID and CommonVoiceLID at matched sentence-length distributions per language.

### 7.1 Spec
Sampling is configured in evaluation YAML files:

```yaml
sampling:
  # bucket boundaries in WORDS (inclusive lower, exclusive upper)
  length_buckets:
    short:  [25, 35]
    medium: [35, 50]
    long:   [50, 100]

  # samples per bucket per language (default for all languages)
  per_language:
    short:  100
    medium: 100
    long:   100

  # what to do if a language has fewer samples than requested in a bucket
  insufficient_data: "include_what_exists"   # one of: "skip_language", "skip_bucket", "include_what_exists", "error"

  # optional: per-language overrides
  overrides:
    mr:
      short: 50
    yo: { short: 30, medium: 30, long: 30 }

  # random seed for reproducibility
  seed: 42
```

### 7.2 Word-counting
- Default tokenizer: unicode-aware whitespace split via `regex` package (`\p{L}+`).
- For languages without whitespace word boundaries (Chinese `zh*`, Japanese `ja`, Thai `th`, Khmer `km`, Lao `lo`, Burmese `my`, Tibetan `bo`), word-counting via whitespace gives nonsense. For these:
  - Apply a **per-language scaling factor** that maps character count → approximate word count, defined in-code in `src/language_id/data/length_buckets.py` (constant `_CHARS_PER_WORD_OVERRIDES`). Reasonable defaults: ~1.5 chars/word for zh/ja, ~5 chars/word for th/km/lo/my/bo.
  - Implementer should expose this as a clean function `count_words(text: str, lang_code: str) -> int` in `data/length_buckets.py`.
- Document this clearly in the architecture docs.

### 7.3 Stratified sampler
`data/sampling.py` exposes:

```python
def length_stratified_sample(
    df: pd.DataFrame,
    text_col: str,
    lang_col: str,
    config: SamplingConfig,
) -> pd.DataFrame:
    """
    Returns a subset of `df` containing, for each language, up to
    `per_language[bucket]` samples drawn from each length bucket.
    Adds a `length_bucket` column ("short" | "medium" | "long") and a
    `word_count` column to the returned DataFrame.
    """
```

Sampling is deterministic given `config.seed`.

### 7.4 Reporting alignment
Length-bucketed metrics in `metrics/sliced.py` use the **same bucket definitions** as sampling. If the user changes the YAML, both sampling and reporting reflect the new buckets. This is the single source of truth for length analysis.

---

## 8. Language code normalization

- All loaders normalize to BCP-47 at load time.
- A versioned mapping table (`src/language_id/languages/_mapping_overrides.yaml`) handles known mismatches between datasets and model outputs.
- LLM free-text outputs are parsed by `parsing/llm_output.py`, which:
  1. Strips common prefixes/suffixes ("The language is...", "Language:", trailing punctuation)
  2. Attempts `langcodes.find()` (handles "Spanish", "Castilian", "español", etc.)
  3. Falls back to a hand-written alias table
  4. Logs unparseable outputs to a dedicated W&B table for later inspection
- The full reconciliation table is rendered into `docs/language-code-reconciliation.md` from the YAML at docs-build time.

---

## 9. Metrics

### 9.1 Primary
- **Macro-F1** (weights all languages equally — surfaces inequity)

### 9.2 Secondary
- Per-language F1
- Confusion matrix (full and zoomed-in on confusable pairs)
- Length-bucketed accuracy (`short`, `medium`, `long` — same buckets as sampling)
- Per-script-family F1 (ISO 15924 grouping)
- Per-resource-tier F1 (`high` / `mid` / `low` from `configs/resource_tiers.yaml`)
- Confusable-pair accuracy (Hindi/Urdu, sr/hr/bs, id/ms, nb/nn, etc. — list configurable)
- Calibration (ECE + reliability diagrams) where the model exposes probabilities

### 9.3 LLM-specific
- Unparseable response rate (per model, per language)
- Mean prediction latency (informational)

---

## 10. CLI design (Typer)

Flat command structure:

```
language-id eval        --model <id> --dataset <id> [--config <path>] [--limit N]
language-id train       --model <id> --dataset <id> [--config <path>]
language-id add-language --code <bcp47> --data <path> [--config <path>]
language-id report      [--from results/] [--to docs/]
```

- `eval` runs a single (model, dataset) evaluation and writes predictions to `results/predictions/<model>_<dataset>_<timestamp>.parquet` and metrics to `results/metrics/`.
- `train` runs training, logs to W&B, saves model artifacts to `results/models/`.
- `add-language` is the Experiment 4 entry point — wraps the data-efficiency loop from Experiment 3.
- `report` regenerates figures and markdown tables from `results/` into `docs/figures/` and `docs/results.md`.

Experiment scripts (`src/language_id/experiments/expN_*.py`) are thin orchestrators that call the same underlying functions as the CLI.

---

## 11. Resource tier classification

Computed once from CommonVoiceLID line counts. Default thresholds in `configs/resource_tiers.yaml`:

```yaml
thresholds:
  high: 1000000      # ≥1M lines
  mid:  10000        # 10k–1M lines
  low:  0            # <10k lines
output_path: src/language_id/languages/_tiers.json   # regenerated by `language-id compute-tiers`
```

A small CLI subcommand `language-id compute-tiers` regenerates the JSON. The full per-language tier table is rendered into the docs as an appendix.

---

## 12. Outputs and reporting

### 12.1 Local artifacts
- `results/predictions/`: parquet, one file per (model, dataset, run)
- `results/metrics/`: JSON, summary metrics per run
- `results/models/`: trained model artifacts (sklearn pickles, LoRA adapters)
- `results/runs.jsonl`: run registry (model, dataset, config hash, seed, W&B URL, timestamps)

### 12.2 W&B
- One project per experiment (`lid-bench-exp1`, `lid-bench-exp2`, etc.)
- Run naming convention: `{experiment}-{model}-{dataset}-{seed}`
- Logged: metrics, confusion matrices, per-language F1 tables, prompt/config artifacts, sklearn plots via `wandb.sklearn.plot_classifier`

### 12.3 Docs
- `language-id report` reads from `results/` and writes:
  - PNG + SVG figures to `docs/figures/`
  - Markdown tables embedded in `docs/results.md`
- Figures and the regenerated `results.md` are **committed to the repo** — mkdocs build is fully static.
- GH Actions workflow (`.github/workflows/docs.yaml`) builds and deploys to GH Pages on push to `main`.

---

## 13. Pre-commit configuration

`.pre-commit-config.yaml` runs `ruff` (lint + format) and `ty` (type check) on commit. No other hooks. Configure both to be strict but pragmatic — type-check `src/`, don't bother with `experiments/` orchestration scripts.

---

## 14. `pyproject.toml` essentials

Project metadata, Python ≥3.12. Dependency groups:

- **Core**: `pandas`, `pydantic`, `pydantic-settings`, `pyyaml`, `typer`, `langcodes`, `regex`, `wandb`, `datacollective`
- **LLMs**: `any-llm`
- **Standard**: `langdetect`, `fasttext-langdetect` (for GlotLID/NLLB-LID loading — use `huggingface_hub` to fetch weights), `transformers`, `torch`
- **Training**: `scikit-learn`, `transformers`, `peft` (for LoRA), `accelerate`, `datasets`
- **Plotting**: `matplotlib`, `plotly`
- **Dev**: `ruff`, `ty`, `pre-commit`, `mkdocs`, `mkdocs-material`

Build system: hatchling or uv's default. Lock with `uv.lock`.

---

## 15. Reproducibility requirements

Every run records:
- Exact model snapshot ID (locked at experiment start)
- Prompt template hash
- Config hash
- Random seeds (data sampling, training, evaluation)
- Dataset version / load timestamp
- Git commit
- W&B run URL

Stored in `results/runs.jsonl` and the W&B run config.

---

## 16. Open items for implementer

1. **Marma loader name** in `datacollective` — use the obvious identifier; flag if not found.
2. **NLLB-200 LID weights** — confirm the canonical HF repo at implementation time.
3. **Word-count overrides** for non-spaced scripts — defaults provided in §7.2; tune if empirical results look off.
4. **Confusable-pair list** — start with the suggested pairs (Hindi/Urdu, sr/hr/bs, id/ms, nb/nn) and extend as confusion-matrix analysis reveals new ones.

---

## 17. Build and CI

- `.github/workflows/docs.yaml`: on push to `main`, install dependencies via `uv`, run `mkdocs build`, deploy to `gh-pages` branch.
- No test workflow (tests are out of scope).
- Optional: a `lint.yaml` workflow that runs `ruff` and `ty` on PRs. Not required but nice to have.

---

## 18. Suggested implementation order

1. `pyproject.toml`, `.pre-commit-config.yaml`, `mkdocs.yaml`, GH workflow — get scaffolding right first.
2. `data/loaders.py` + `data/normalization.py` + `languages/codes.py` — get datasets loading and normalized.
3. `data/length_buckets.py` + `data/sampling.py` — get length-stratified sampling working with a small example.
4. `models/base.py` + one standard model (`langdetect_model.py`) — prove the protocol end-to-end.
5. `metrics/core.py` + `metrics/sliced.py` + a minimal CLI `eval` command — run langdetect on a sampled CommonLID slice, log to W&B, see metrics.
6. Add remaining standard models, then LLMs (with caching).
7. Trained classifiers (LogReg, NB, XLM-R with LoRA).
8. Experiment 3 (Marma curve), Experiment 4 (add-language CLI), Experiment 5 (cross-benchmark).
9. `reporting/figures.py` + `reporting/tables.py` + `language-id report`.
10. Docs polish.

