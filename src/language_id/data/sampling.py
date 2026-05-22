"""Length-stratified sampling (spec §7)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from language_id.data.length_buckets import assign_bucket, count_words

InsufficientDataPolicy = Literal[
    "skip_language", "skip_bucket", "include_what_exists", "error"
]


class SamplingConfig(BaseModel):
    """Configuration for length-stratified sampling (spec §7.1)."""

    length_buckets: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
            "short": (25, 35),
            "medium": (35, 50),
            "long": (50, 100),
        }
    )
    per_language: dict[str, int] = Field(
        default_factory=lambda: {"short": 100, "medium": 100, "long": 100}
    )
    insufficient_data: InsufficientDataPolicy = "include_what_exists"
    overrides: dict[str, dict[str, int]] = Field(default_factory=dict)
    seed: int = 42


def length_stratified_sample(
    df: pd.DataFrame,
    text_col: str,
    lang_col: str,
    config: SamplingConfig,
) -> pd.DataFrame:
    """Return a length-stratified subset of `df`.

    For each language, draws up to `per_language[bucket]` rows from each length
    bucket. Sampling is deterministic given `config.seed`.

    Adds two columns to the returned DataFrame:
        - `length_bucket`: bucket name ("short" | "medium" | "long")
        - `word_count`: integer word count (per spec §7.2)
    """
    rng = np.random.default_rng(config.seed)

    work = df.copy()
    work["word_count"] = [
        count_words(t, l) for t, l in zip(work[text_col], work[lang_col], strict=True)
    ]
    work["length_bucket"] = work["word_count"].map(
        lambda wc: assign_bucket(wc, config.length_buckets)
    )
    work = work[work["length_bucket"].notna()].reset_index(drop=True)

    out_frames: list[pd.DataFrame] = []
    for lang in sorted(work[lang_col].unique()):
        sub = work[work[lang_col] == lang]
        lang_overrides = config.overrides.get(lang, {})
        targets = {
            bucket: lang_overrides.get(bucket, config.per_language.get(bucket, 0))
            for bucket in config.length_buckets
        }
        counts = sub.groupby("length_bucket").size().to_dict()
        insufficient = [
            b for b, t in targets.items() if t > 0 and counts.get(b, 0) < t
        ]

        if insufficient and config.insufficient_data == "error":
            raise ValueError(
                f"lang={lang}: buckets {insufficient} have insufficient data "
                f"(have {counts}, want {targets})"
            )
        if insufficient and config.insufficient_data == "skip_language":
            continue

        for bucket, target in targets.items():
            if target <= 0:
                continue
            available = sub[sub["length_bucket"] == bucket]
            n_avail = len(available)
            if n_avail == 0:
                continue
            if n_avail < target:
                if config.insufficient_data == "skip_bucket":
                    continue
                sampled = available
            else:
                idx = rng.choice(n_avail, size=target, replace=False)
                sampled = available.iloc[idx]
            out_frames.append(sampled)

    if not out_frames:
        return work.iloc[0:0]
    return pd.concat(out_frames, ignore_index=True)
