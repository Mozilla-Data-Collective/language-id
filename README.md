# Text Language Identification with Mozilla Data Collective

Text Language Identification (LID) is still an unresolved problem for most languages in the world. Significant progress has been made using methods character n-gram models (langdetect, fastText) and neural approaches (XLM-R, GlotLID), however performance is disproportionately distributed to favour high-resource languages such as English.

This project provides a CLI and a set of notebooks to **benchmark existing LID models** (langdetect, GlotLID, NLLB-LID, and LLMs) as well as to **train your own language detector** on datasets from the [Mozilla Data Collective](https://mozilladatacollective.com/?utm_source=webinar&utm_medium=online-event&utm_campaign=common-crawl-event) platform, such as [CommonLID](https://mozilladatacollective.com/datasets/cmp5c60at015po007bbql6h3s?utm_source=webinar&utm_medium=online-event&utm_campaign=common-crawl-event), [Common Voice LID](https://mozilladatacollective.com/datasets/cmj8ddapc02c8mb07l6wyr882?utm_source=webinar&utm_medium=online-event&utm_campaign=common-crawl-event), [Ladino](https://mozilladatacollective.com/datasets/cmo1ks4zv004enr07la1rkr9x?utm_source=webinar&utm_medium=online-event&utm_campaign=common-crawl-event) and many others.

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/Mozilla-Data-Collective/language-id.git
cd language-id
uv sync
```

## Quick Start

1. **Get your API key** from the [Mozilla Data Collective platform](https://mozilladatacollective.com/?utm_source=webinar&utm_medium=online-event&utm_campaign=common-crawl-event) and add it to a `.env` file in the repo root:

```
MDC_API_KEY=your-api-key-here
```

2. **Evaluate a model** on the CommonLID benchmark:

```bash
uv run language-id eval --eval-model glotlid --dataset commonlid --n 200
```

3. **Train your own single-language detector** for any language with a text corpus on MDC:

```bash
uv run language-id train --dataset your-dataset-id --lang lad
```
_Note: See our guide ["bring-your-own-dataset"](https://mozilla-data-collective.github.io/language-id/bring-your-own-dataset/) for details on how to parse it properly in the codebase._

Run `uv run language-id --help` to see all commands and options.

## Notebooks

| Notebook | Description                                                                                                        |
|---|--------------------------------------------------------------------------------------------------------------------|
| [run-model-evaluation](src/language_id/notebooks/run-model-evaluation.ipynb) | Benchmark off-the-shelf LID tools and LLMs (zero- and few-shot) side by side on CommonLID or Common Voice LID.     |
| [train-single-language-detector](src/language_id/notebooks/train-single-language-detector.ipynb) | Train a specialist LID detector for a single language and compare it against off-the-shelf baselines. |
| [train-and-evaluate-local-lid](src/language_id/notebooks/train-and-evaluate-local-lid.ipynb) | From-scratch tutorial: train a full multi-class character n-gram + Naive Bayes detector locally, evaluate it, and analyse its errors. |
| [add-new-language-to-lid](src/language_id/notebooks/add-new-language-to-lid.ipynb) | Add support for a new (often low-resource) language by folding your own corpus into the training and evaluation data. |
| [compare-saved-runs](src/language_id/notebooks/compare-saved-runs.ipynb) | Helper notebook to compare experimental results (saved runs) and create tables and graphs for further analysis.    |

## For more details, visit [our docs](https://mozilla-data-collective.github.io/language-id/)

## License

This project is released under [MPL (Mozilla Public License) 2.0](./LICENSE).
