# Models

Run `language-id models` to list every model name accepted by `eval --model`. There are two families: standard LID tools that run locally, and LLMs served via the [Together](https://www.together.ai/) API.

## Standard "off-the-shelf" LID tools

These run fully locally on CPU, have low RAM requirements and need no API key:

| Name | Description |
|---|---|
| `langdetect` | Port of Google's classic character n-gram language detector (~55 languages). |
| `glotlid` | [GlotLID](https://huggingface.co/cis-lmu/glotlid) — a fastText model covering ~2000 language varieties, downloaded from Hugging Face on first use. |
| `nllb-lid` | The fastText LID model from Meta's NLLB project (~200 languages), downloaded from Hugging Face on first use. |

## LLMs 

LLMs are prompted to classify the language of a sentence and can be run zero-shot or few-shot (`--shot 1..10`); few-shot examples are held out from the evaluation set. They require `TOGETHER_API_KEY`. The current registry includes models such as `gpt-oss-120b`, `gpt-oss-20b`, `gemma`, `llama`, and `minimax-m27`. `language-id models` always shows the up-to-date list.

### Together AI

Right now, the only way to evaluate LLMs is through the [Together API](https://www.together.ai/), which serves many open and closed models with a simple interface. The API client is implemented in `src/language_id/models/together.py` and can be easily extended to new models.

### WIP: Otari

The [Otari](https://otari.ai/) client currently a work in progress, but it implements an identical interface to the together.py client. Otari is able to provide a single interface to connect to any local or hosted model, served through Ollama or other 3rd party providers such as OpenAI, Anthropic, Mistral, etc. Check out [otari.py](https://github.com/Mozilla-Data-Collective/language-id/blob/main/src/language_id/models/otari.py) for more details.


## Trainable detectors

`language-id train` fits a *single-language* detector (target language vs. everything else) rather than a multi-way classifier:

| Method        | Description                                                                                                                                                                   |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `naive_bayes` | TF-IDF character n-grams + naive bayes. Fast, CPU-only.                                                                                                                       |
| `logreg`      | TF-IDF character n-grams + logistic regression. Fast, CPU-only.                                                                                                               |
| `llm`         | Fine-tune any Hugging Face model with a sequence-classification head (default `Qwen/Qwen3-0.6B`). Swap `--hf-model-id` to try another base model. Needs the `finetune` extra. |

See the [CLI reference](cli.md) and the [train-single-language-detector notebook](notebooks.md) for usage.
