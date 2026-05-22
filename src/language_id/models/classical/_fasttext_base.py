"""Shared logic for fastText-based LID models (GlotLID, NLLB-LID).

The `fasttext` Python package is NOT in default dependencies. Install it (or
`fasttext-wheel` for pre-built wheels) to enable these models.
"""

from __future__ import annotations

from typing import Any

from language_id.languages.codes import to_bcp47
from language_id.models.base import LIDPrediction


def fasttext_label_to_bcp47(label: str) -> str:
    """Convert a fastText label (e.g. "__label__eng_Latn") to BCP-47."""
    return to_bcp47(label.removeprefix("__label__").replace("_", "-"))


class FastTextLIDBase:
    """Base class for fastText-backed LID models.

    Subclasses set `name`, `version`, and the default `hf_repo` / `hf_filename`.
    Weights are downloaded from HuggingFace Hub on first call.
    """

    name: str = "fasttext"
    version: str = "TBD"

    def __init__(
        self,
        hf_repo: str,
        hf_filename: str = "model.bin",
        top_k: int = 1,
    ) -> None:
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self.top_k = top_k
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import fasttext
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires the `fasttext` package "
                "(not in default deps; install `fasttext` or `fasttext-wheel`)."
            ) from e
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)
        self._model = fasttext.load_model(path)

    def predict(self, text: str) -> LIDPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        self._ensure_loaded()
        # fastText breaks on embedded newlines.
        cleaned = [t.replace("\n", " ").strip() for t in texts]
        labels_list, probs_list = self._model.predict(cleaned, k=self.top_k)
        return [
            LIDPrediction(
                lang_code=fasttext_label_to_bcp47(labels[0]),
                confidence=float(probs[0]),
                raw_output=f"{labels[0]} ({probs[0]:.4f})",
            )
            for labels, probs in zip(labels_list, probs_list, strict=True)
        ]
