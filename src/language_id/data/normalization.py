import pandas as pd

from language_id.languages.codes import to_bcp47


def normalize_lang_column(df: pd.DataFrame, col: str = "lang") -> pd.DataFrame:
    """Return a copy of `df` with `col` normalized to canonical BCP-47.

    Uses `languages.codes.to_bcp47` plus the override table in
    `src/language_id/languages/_mapping_overrides.yaml`.
    """
    if col not in df.columns:
        raise KeyError(f"column {col!r} not in DataFrame; have {list(df.columns)}")
    out = df.copy()
    out[col] = out[col].astype(str).map(to_bcp47)
    return out
