import pandas as pd
from datacollective import load_dataset

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
