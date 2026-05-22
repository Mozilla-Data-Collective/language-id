"""GlotLID baseline. Weights via `huggingface_hub`. Spec §3 Experiment 1."""

from __future__ import annotations

from language_id.models.classical._fasttext_base import FastTextLIDBase


class GlotLIDModel(FastTextLIDBase):
    """GlotLID v3 — fastText-based LID covering 2000+ languages.

    Requires the `fasttext` package (not in default deps; see `_fasttext_base`).
    """

    name = "glotlid"
    version = "v3"

    def __init__(
        self,
        hf_repo: str = "cis-lmu/glotlid",
        hf_filename: str = "model_v3.bin",
        top_k: int = 1,
    ) -> None:
        super().__init__(hf_repo=hf_repo, hf_filename=hf_filename, top_k=top_k)
