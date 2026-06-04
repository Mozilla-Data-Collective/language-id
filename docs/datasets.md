# Datasets

All datasets are accessed through the [Mozilla Data Collective](https://mozilladatacollective.com/) platform via the [`datacollective`](https://github.com/Mozilla-Data-Collective/datacollective-python) Python SDK, which requires `MDC_API_KEY` (see [Getting started](getting-started.md)). Before accessing any dataset, make sure you have read and agreed to that dataset's conditions and licensing terms. Downloads are cached locally, so each dataset is fetched only once.

## Built-in datasets

These two are available by name in `language-id eval --dataset ...`:

| Name | Role | Description                                                                                                                                                                                                |
|---|---|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `commonlid` | Evaluation | **CommonLID** — an evaluation-only LID benchmark built from Common Crawl, with over 300,000 examples covering more than 100 languages.                                                                     |
| `commonvoice_lid` | Training + evaluation | **Common Voice LID** (CV-LID) — text from Mozilla Common Voice projects covering over 300 languages (~20M rows), with train/dev/test splits. Also the source of negative examples for `language-id train`. |

## Bring-your-own-dataset

Any MDC dataset with a text column and a language column can be evaluated by passing its dataset ID:

> See the dedicated [Bring your own dataset](bring-your-own-dataset.md) guide for the two ways to load an MDC dataset (`load_dataset` vs. `download_dataset` + custom parsing) and full evaluate/train examples.

```bash
uv run language-id eval --eval-model glotlid --dataset your-dataset-id \
    --lang-col tag --text-col sentence
```


### Single-language corpora

For a corpus that is entirely in one language (so it has no language column), pass `--ground-truth-language`: every row's gold label becomes that language, and only the text column is needed.

```bash
uv run language-id eval --eval-model glotlid --dataset your-dataset-id \
    --ground-truth-language lad
```

Single-language corpora are also the input to `language-id train`, which pairs them with Common Voice LID negatives to train a detector — see the [CLI reference](cli.md).

## Language code normalization

Datasets and models label languages inconsistently — ISO-639-1 (`en`), ISO-639-3 (`eng`), BCP-47 tags (`en-US`), or plain names (`English`). Every loader and every model prediction is normalized to **ISO-639-3** (via [`langcodes`](https://pypi.org/project/langcodes/) plus a project-level mapping in `src/language_id/lang_codes_mapping.csv`), so metrics are always computed on a consistent label space. Anywhere the CLI accepts a language (`--langs`, `--lang`, `--ground-truth-language`), any code or name form works.
