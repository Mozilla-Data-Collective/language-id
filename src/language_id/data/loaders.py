import pandas as pd
from datacollective import load_dataset

from language_id.data.normalization import normalize_lang_column

COMMON_LID_DATASET_ID = "commonlid-622f6f75"
COMMON_VOICE_LID_DATASET_ID = "mozilla-common-voice-text-language-ident-b1b3aae0"
MARMA_DATASET_ID = "cmnxhxm1i0090nu071ba8o4sg"


def load_commonlid() -> pd.DataFrame:
    """Load CommonLID.
    https://mozilladatacollective.com/datasets/cmp5c60at015po007bbql6h3s

    Returns a DataFrame with `lang` normalized to BCP-47."""
    df = load_dataset(COMMON_LID_DATASET_ID)
    return normalize_lang_column(df, col="tag")


def load_commonvoice_lid(split: str | None = None) -> pd.DataFrame:
    """Load CommonVoiceLID.
    https://mozilladatacollective.com/datasets/cmj8ddapc02c8mb07l6wyr882

    Args:
        split: One of "train", "dev", "test", or None for the full dataset.

    Returns a DataFrame with columns: id, sentence, lang (BCP-47), sentence_domain, source, style, split.
    """
    df = load_dataset(COMMON_VOICE_LID_DATASET_ID)
    df = normalize_lang_column(df, col="lang")
    if split is None:
        return df
    return df[df["split"] == split].reset_index(drop=True)


def load_marma() -> pd.DataFrame:
    """Load the Marma corpus via `datacollective`.
    """
    df = load_dataset(MARMA_DATASET_ID)
    return normalize_lang_column(df, col="lang")


def load_dataset_by_id(dataset_id: str) -> pd.DataFrame:
    """Load any `datacollective` dataset by ID (bring-your-own-data).

    This is how users supply their own corpus / language: exactly the same
    path as CommonLID / CommonVoiceLID / Marma — pass a Mozilla Data Collective
    dataset ID. The dataset is expected to carry a `lang` column (normalized to
    BCP-47) and a text column (`text` or `sentence`, see `resolve_text_col`).
    """
    df = load_dataset(dataset_id)
    return normalize_lang_column(df, col="lang")


def resolve_text_col(df: pd.DataFrame) -> str:
    """Return the text column for an arbitrary dataset (`text` or `sentence`)."""
    for col in ("text", "sentence"):
        if col in df.columns:
            return col
    raise KeyError(
        f"dataset has no 'text' or 'sentence' column; got {list(df.columns)}"
    )
