# Getting started

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)

## Installation

```bash
git clone https://github.com/Mozilla-Data-Collective/language-id.git
cd language-id
uv sync
```

### Optional extras

| Extra | Install | What it adds |
|---|---|---|
| `finetune` | `uv sync --extra finetune` | `torch`, `transformers`, `datasets`, `accelerate` — needed to fine-tune a Hugging Face model as a single-language detector (`language-id train --train-model llm`). |
| `dev` | `uv sync --extra dev` | Linting (`ruff`), type checking (`ty`), `pre-commit`, and the MkDocs documentation toolchain. |

## API keys

Create a `.env` file in the repository root:

```
MDC_API_KEY=your-api-key-here
TOGETHER_API_KEY=your-api-key-here
```

- **`MDC_API_KEY`** (required) — used to download datasets from the Mozilla Data Collective platform. Get yours from the [MDC dashboard](https://mozilladatacollective.com/api-reference). Before accessing any dataset, make sure you have read and agreed to that dataset's conditions and licensing terms.
- **`TOGETHER_API_KEY`** (optional) — only needed to evaluate LLMs served via [Together](https://www.together.ai/) (see [Models](models.md)). The standard LID tools run fully locally without it.

**_WIP:_** You can also use the Otari SDK client which lets you connect to both local models (e.g. through Ollama) and hosted 3rd party ones (e.g. Together AI, OpenAI, Anthropic, Mistral, etc) with the same interface. It follows the same exact logic as the together.py client, however, right now, its not integrated in the CLI. Check out [otari.py](https://github.com/Mozilla-Data-Collective/language-id/blob/main/src/language_id/eval_models/otari.py) for more details.

Alternatively, export them in your shell:

```bash
export MDC_API_KEY=your-api-key-here
```

## Verify the installation

```bash
uv run language-id eval-models
```

This lists every eval model name the CLI can evaluate.

!!! note "Eval models vs. train models"
    Throughout the CLI and docs, **eval models** (`eval --eval-model`) are the ready-made, off-the-shelf models you *evaluate* (standard tools + LLMs), while **train models** (`train --train-model`) are the kinds of models you *train* yourself (`naive_bayes`, `logreg`, or a fine-tuned HF `llm`). See [Models](models.md).

Then run a small first evaluation:

```bash
uv run language-id eval --eval-model langdetect --dataset commonlid --n 50
```

The first run downloads the dataset from MDC; subsequent runs reuse the local copy. Results are printed to the terminal and saved under `results/` — see the [CLI reference](cli.md) for details.
