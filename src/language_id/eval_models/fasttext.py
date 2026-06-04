from typing import Any

from tqdm import tqdm

from language_id.eval_models.base import LIDPrediction
from language_id.lang_codes_mapping import to_iso3


class FastTextLIDBase:
    """Base class for fastText-backed LID models: GlotLID and NLLB-200 LID.
    Weights download from the HuggingFace Hub on first call. Requires the
    `fasttext` module (provided by the `fasttext-predict` dependency).
    fastText emits labels like "__label__eng_Latn"; `to_iso3` keeps the code.
    """

    name = "fasttext"

    def __init__(self, hf_repo: str, hf_filename: str = "model.bin") -> None:
        self.hf_repo = hf_repo
        self.hf_filename = hf_filename
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import fasttext
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo_id=self.hf_repo, filename=self.hf_filename)
        self._model = fasttext.load_model(path)

    def predict(self, text: str) -> LIDPrediction:
        self._ensure_loaded()
        # fastText breaks on embedded newlines.
        cleaned = text.replace("\n", " ").strip()
        labels, probs = self._model.predict(cleaned, k=1)
        if not labels:
            return LIDPrediction(lang_code="und", confidence=None, raw_output="")
        return LIDPrediction(
            lang_code=to_iso3(labels[0].removeprefix("__label__")),
            confidence=float(probs[0]),
            raw_output=f"{labels[0]} ({probs[0]:.4f})",
        )

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in tqdm(texts, desc=self.name, unit="text")]


class GlotLIDModel(FastTextLIDBase):
    """GlotLID v3: fastText LID covering 2000+ languages."""

    name = "glotlid"

    def __init__(
        self,
        hf_repo: str = "cis-lmu/glotlid",
        hf_filename: str = "model_v3.bin",
    ) -> None:
        super().__init__(hf_repo=hf_repo, hf_filename=hf_filename)


class NLLBLIDModel(FastTextLIDBase):
    """NLLB-200 fastText LID head."""

    name = "nllb-lid"

    def __init__(
        self,
        hf_repo: str = "facebook/fasttext-language-identification",
        hf_filename: str = "model.bin",
    ) -> None:
        super().__init__(hf_repo=hf_repo, hf_filename=hf_filename)
