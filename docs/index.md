# Text Language Identification with Mozilla Data Collective

Text Language Identification (LID) is still an unresolved problem for most languages in the world. Significant progress has been made using methods character n-gram models (langdetect, fastText) and neural approaches (XLM-R, GlotLID), however performance is disproportionately distributed to favour high-resource languages such as English.

This project provides tools to measure and close that gap using datasets from the [Mozilla Data Collective](https://mozilladatacollective.com/) platform:

- **Benchmark existing LID models** standard "off-the-shelf" Python tools (langdetect, GlotLID, NLLB-LID) and LLMs  on the CommonLID and Common Voice LID datasets, or on any MDC dataset you bring.
- **Train your own language detector** for a specific language from a plain text corpus, using either a Naive Bayes classifier (no GPU or high RAM requirement) or a fine-tuned Hugging Face model (requires GPU).
- **Compare results** across models and runs with per-language metrics, heatmaps, and disagreement analysis.

## Where to go next

- [Getting started](getting-started.md): Quick project overview and installation instructions.
- [CLI reference](cli.md): The `language-id` command: `eval`, `train`, and `models`.
- [Notebooks](notebooks.md): Interactive walkthroughs of evaluation, comparison, and training.
- [Bring-your-own-dataset](bring-your-own-dataset.md): Guide on how to evaluate and train on any MDC text dataset.
- [Datasets](datasets.md): Information on the datasets used.
- [Models](models.md): The available LID models and how to add more.
