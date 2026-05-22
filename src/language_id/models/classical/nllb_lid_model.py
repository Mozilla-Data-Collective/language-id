"""NLLB-200 LID head. Spec §3 Experiment 1; §16 open item — confirm HF repo."""

from __future__ import annotations

from language_id.models.classical._fasttext_base import FastTextLIDBase


class NLLBLIDModel(FastTextLIDBase):
    """NLLB-200 fastText LID head.

    Requires the `fasttext` package (not in default deps; see `_fasttext_base`).
    Default repo `facebook/fasttext-language-identification` — confirm at run time.
    """

    name = "nllb-lid"
    version = "TBD"

    def __init__(
        self,
        hf_repo: str = "facebook/fasttext-language-identification",
        hf_filename: str = "model.bin",
        top_k: int = 1,
    ) -> None:
        super().__init__(hf_repo=hf_repo, hf_filename=hf_filename, top_k=top_k)
