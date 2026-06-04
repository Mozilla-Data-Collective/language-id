# Bring your own dataset

Any text dataset hosted on the [Mozilla Data Collective](https://mozilladatacollective.com/) platform can be plugged into this project for evaluation or training, given that the license and terms & conditions approve of such usage. There are two ways to get an MDC dataset into the pipeline, depending on its format:

| Path | SDK function | When to use                                                                                                                                                                                                  |
|---|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Load directly](#path-1-load-the-dataset-directly) | `load_dataset` | Certain datasets can be parsed directly into a pandas DataFrame (e.g. a TSV with a text column). See `load_dataset_by_id` in `src/language_id/data.py`.                                                      |
| [Download + custom parsing](#path-2-download-the-archive-and-parse-it-yourself) | `download_dataset` | The dataset's format isn't supported by `load_dataset`, so you download the raw archive and write your own parsing logic. See `download_and_load_single_language_text_dataset` in `src/language_id/data.py`. |

Whichever path you take, the convention in this codebase is that every loader returns a DataFrame with:

- a **`sentence`** column: the text to identify, and
- a **`lang`** column: the gold language label, normalized to **ISO-639-3** with `to_iso3` (see [Datasets](datasets.md#language-code-normalization)).

Once your data is in that shape, every downstream piece (evaluation, training, metrics, plots) works unchanged.

## Path 1: Load the dataset directly

If the SDK's `load_dataset` supports the dataset's format, `load_dataset_by_id` is all you need: it loads the DataFrame, renames your columns to the project conventions, and normalizes the language labels:

```python
from language_id.data import load_dataset_by_id

# A multilingual dataset whose language column is named `tag`
# and whose text column is named `text`.
df = load_dataset_by_id("your-dataset-id", lang_col_name="tag", text_col_name="text")
df.head()  # columns: sentence, lang (ISO-639-3), ...
```

If the dataset is entirely in **one language** (so it has no language column), use `load_single_language_dataset` instead and supply the language once and it will be applied in every row automatically:

```python
from language_id.data import load_single_language_dataset

df = load_single_language_dataset("your-dataset-id", ground_truth_language="lad")
```

### Evaluate it

From the CLI, this path corresponds to passing the dataset ID and column names directly:

```bash
# Multilingual dataset with its own language column
uv run language-id eval --model glotlid --dataset your-dataset-id \
    --lang-col tag --text-col text

# Single-language dataset: every row's gold label is the given language
uv run language-id eval --model glotlid --dataset your-dataset-id \
    --ground-truth-language lad
```

Or in Python, pass the DataFrame straight to `evaluate`:

```python
from language_id.evaluate import evaluate
from language_id.models import get_model

overall, per_lang, predictions = evaluate(df, get_model("glotlid"))
print(overall)
```

## Path 2: Download the archive and parse it yourself

Some datasets can't be used with `load_dataset` as of today. For those, use the SDK's `download_dataset` to fetch the raw archive, then write the appropriate parsing logic.

`download_and_load_single_language_text_dataset` in `src/language_id/data.py` is the reference implementation of this path: it downloads the archive, extracts it, reads every `.txt` file (one sentence per line) into the standard `sentence`/`lang` DataFrame, and drops empty lines:

```python
from language_id.data import download_and_load_single_language_text_dataset

df = download_and_load_single_language_text_dataset(
    "your-dataset-id", ground_truth_language="lad"
)
df.head()  # columns: sentence, lang (ISO-639-3)
```

### Evaluate or train with it

A DataFrame built this way feeds directly into `evaluate`, exactly as in Path 1:

```python
from language_id.evaluate import evaluate
from language_id.models import get_model

overall, per_lang, predictions = evaluate(df, get_model("glotlid"))
```

For **training**, `language-id train` already uses this path under the hood: `build_training_data` calls `download_and_load_single_language_text_dataset` to load your corpus as positives and pairs it with Common Voice LID negatives. So for a single-language `.txt`-archive corpus, training is one command:

```bash
uv run language-id train --dataset your-dataset-id --lang lad --method naive_bayes
```

If your corpus needs custom parsing, load it with your own function (as above) and use the training building blocks directly:

```python
from language_id.train import train_naive_bayes, evaluate_detector

# train_df needs `sentence` + `label` columns, where `label` is the target
# ISO-639-3 code for positives and "other" for negatives — see
# `build_training_data` in src/language_id/train.py for the reference recipe.
model = train_naive_bayes(train_df)
scores = evaluate_detector(model, test_df, "lad")
print(scores)
```

See the [CLI reference](cli.md) for every `eval`/`train` option, and the [train-single-language-detector notebook](notebooks.md) for an end-to-end interactive walkthrough.