import tarfile
from pathlib import Path

import pandas as pd
from datacollective import download_dataset, load_dataset

from language_id.lang_codes_mapping import to_iso3

COMMON_LID_DATASET_ID = "commonlid-622f6f75"
COMMON_VOICE_LID_DATASET_ID = "mozilla-common-voice-text-language-ident-b1b3aae0"

LANG_COLUMN_NAME = "lang"
TEXT_COLUMN_NAME = "sentence"

# Every loader returns a DataFrame with a `text` column and a `lang` column normalized to ISO-639-3.
def load_commonlid() -> pd.DataFrame:
    """Load CommonLID (evaluation set). Columns: text, lang (ISO-639-3), line_id."""
    df = load_dataset(COMMON_LID_DATASET_ID)
    df = df.rename(columns={"tag": LANG_COLUMN_NAME})
    df = df.rename(columns={"text": TEXT_COLUMN_NAME})
    df["lang"] = df["lang"].astype(str).map(to_iso3)
    return df


def load_commonvoice_lid(split: str | None = None) -> pd.DataFrame:
    """Load CommonVoiceLID (training-available). Text column is `sentence`.

    `split`: one of "train"/"dev"/"test", or None for the full dataset.
    """
    df = load_dataset(COMMON_VOICE_LID_DATASET_ID)
    df["lang"] = df["lang"].astype(str).map(to_iso3)
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    return df


def load_dataset_by_id(dataset_id: str, lang_col_name: str = "tag", text_col_name: str = "sentence") -> pd.DataFrame:
    """Load any datacollective dataset by ID (bring-your-own-dataset).

    Requires to have a `lang` column (normalized to ISO-639-3) and a
    text column (could be `text` or `sentence`).

    Args:
        dataset_id: The valid Mozilla Data Collective dataset ID or slug.
        lang_col_name: The name of the language identifier column (e.g. 'en', 'English').
        text_col_name: The name of the text column (the actual sentences to identify the language of).
    """
    df = load_dataset(dataset_id)
    df = df.rename(columns={lang_col_name: LANG_COLUMN_NAME})
    df = df.rename(columns={text_col_name: TEXT_COLUMN_NAME})
    df["lang"] = df["lang"].astype(str).map(to_iso3)
    return df


def load_single_language_dataset(
    dataset_id: str,
    ground_truth_language: str,
    text_col_name: str = "sentence",
) -> pd.DataFrame:
    """Load a single-language dataset whose every row shares one known language.

    Use this when the dataset has only a text column and no per-row language
    label: the gold `lang` is supplied once and applied to every row.

    Args:
        dataset_id: The valid Mozilla Data Collective dataset ID or slug.
        ground_truth_language: The language of every row, in any code/name form
            (normalized to ISO-639-3).
        text_col_name: The name of the text column (the sentences to identify).
    """
    df = load_dataset(dataset_id)
    df = df.rename(columns={text_col_name: TEXT_COLUMN_NAME})
    df[LANG_COLUMN_NAME] = to_iso3(ground_truth_language)
    return df


def load_single_language_text_archive(
    dataset_id: str,
    ground_truth_language: str,
    download_directory: str | None = None,
) -> pd.DataFrame:
    """Load a single-language corpus distributed as a `.tar.gz` of `.txt` files.

    Some datasets can't be read with `load_dataset` (for example, they consist of
    a single raw text archive rather than a tabular file). This downloads the archive, extracts
    it, and reads every `.txt` file with one sentence per line into a DataFrame
    with `sentence` and `lang` columns (the language is the same for every row,
    normalized to ISO-639-3).

    Clone/extend this function as you wish to add support for datasets with different formats

    Args:
        dataset_id: The valid Mozilla Data Collective dataset ID or slug.
        ground_truth_language: The language of every row, in any code/name form.
        download_directory: Where to save the archive (defaults to the SDK's path).
    """
    archive_path = download_dataset(dataset_id, download_directory=download_directory)
    extract_dir = Path(str(archive_path).removesuffix(".tar.gz"))
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    lines: list[str] = []
    for txt in sorted(extract_dir.rglob("*.txt")):
        lines.extend(txt.read_text(encoding="utf-8").splitlines())

    df = pd.DataFrame({TEXT_COLUMN_NAME: lines})
    df[LANG_COLUMN_NAME] = to_iso3(ground_truth_language)
    df = df.dropna(subset=[TEXT_COLUMN_NAME])
    df[TEXT_COLUMN_NAME] = df[TEXT_COLUMN_NAME].map(str)
    return df[df[TEXT_COLUMN_NAME].str.strip() != ""].reset_index(drop=True)


def take_fewshot_examples(
    df: pd.DataFrame, shot: int, seed: int = 0
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Pick `shot` demonstration rows for few-shot LLM prompting.

    Examples span distinct languages where possible (better task coverage), are
    chosen deterministically from `seed`, and are removed from the returned
    DataFrame so they never leak into the evaluation set.

    Returns `(examples, remaining)` where `examples` is a list of
    `(text, lang_iso3)` pairs and `remaining` is `df` with those rows dropped.
    """
    if shot <= 0:
        return [], df
    shuffled = df.sample(frac=1.0, random_state=seed)
    picked: list[int] = []
    seen: set[str] = set()
    # Prefer one example per distinct language for variety.
    for idx, lang in shuffled[LANG_COLUMN_NAME].items():
        if len(picked) >= shot:
            break
        if lang not in seen:
            picked.append(idx)
            seen.add(lang)
    # Top up with any remaining rows when there aren't enough languages.
    if len(picked) < shot:
        for idx in shuffled.index:
            if len(picked) >= shot:
                break
            if idx not in picked:
                picked.append(idx)
    examples = [
        (str(df.at[i, TEXT_COLUMN_NAME]), str(df.at[i, LANG_COLUMN_NAME])) for i in picked
    ]
    remaining = df.drop(index=picked).reset_index(drop=True)
    return examples, remaining


def sample(
    df: pd.DataFrame,
    n_per_lang: int,
    langs: list[str] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Return up to `n_per_lang` rows per language (optionally filtered to `langs`).

    `langs` may be given in any code/name form; they're normalized to ISO-639-3.

    Example usage:
        df = load_commonlid()
        df_subset = sample(df, n_per_lang=10, langs=["English", "fra", "deu"], seed=42)

    """
    if langs is not None:
        wanted = {to_iso3(lang) for lang in langs}
        df = df[df["lang"].isin(wanted)]
    parts = [
        g.sample(n=min(n_per_lang, len(g)), random_state=seed)
        for _, g in df.groupby("lang", sort=False)
    ]
    if not parts:
        return df.iloc[0:0].reset_index(drop=True)
    return pd.concat(parts).reset_index(drop=True)
