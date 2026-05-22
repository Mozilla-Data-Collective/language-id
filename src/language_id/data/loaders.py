"""Dataset loaders. All return pandas DataFrames with language codes normalized to BCP-47."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from language_id.data.normalization import normalize_lang_column

COMMON_LID_DATASET_ID = "commonlid-622f6f75"
COMMON_VOICE_LID_DATASET_ID = "mozilla-common-voice-text-language-ident-b1b3aae0"
MARMA_DATASET_ID = "cmnxhxm1i0090nu071ba8o4sg"


def _load(dataset_id: str) -> pd.DataFrame:
    from datacollective import load_dataset

    return load_dataset(dataset_id)


def load_commonlid() -> pd.DataFrame:
    """Load CommonLID. Returns a DataFrame with `lang` normalized to BCP-47."""
    df = _load(COMMON_LID_DATASET_ID)
    return normalize_lang_column(df, col="lang")


def load_commonvoice_lid(split: str | None = None) -> pd.DataFrame:
    """Load CommonVoiceLID.

    Args:
        split: One of "train", "dev", "test", or None for the full dataset.

    Returns a DataFrame with the spec §2.2 columns: id, sentence, lang (BCP-47),
    sentence_domain, source, style, split.
    """
    df = _load(COMMON_VOICE_LID_DATASET_ID)
    df = normalize_lang_column(df, col="lang")
    if split is None:
        return df
    return df[df["split"] == split].reset_index(drop=True)


def load_marma() -> pd.DataFrame:
    """Load the Marma corpus via `datacollective`.
    """
    df = _load(MARMA_DATASET_ID)
    return normalize_lang_column(df, col="lang")


def load_user_language_data(path: Path, default_lang: str) -> pd.DataFrame:
    """Load a user-provided corpus for the `add-language` workflow.

    Accepts either JSONL (`{"text": ..., "lang": ...}` per line; `lang`
    defaults to `default_lang` if absent) or plain text (one example per
    line, all assigned `default_lang`). Detection is suffix-based with a
    JSONL sniff fallback.
    """
    path = Path(path)
    raw = path.read_text()
    suffix = path.suffix.lower()
    first_nonblank = next((line.strip() for line in raw.splitlines() if line.strip()), "")
    looks_like_jsonl = suffix in {".jsonl", ".json"} or first_nonblank.startswith("{")

    rows: list[dict[str, str]] = []
    if looks_like_jsonl:
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({"text": rec["text"], "lang": rec.get("lang") or default_lang})
    else:
        for line in raw.splitlines():
            line = line.strip()
            if line:
                rows.append({"text": line, "lang": default_lang})

    df = pd.DataFrame(rows)
    return normalize_lang_column(df, col="lang")
