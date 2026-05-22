"""langdetect baseline (Python package). Spec §3 Experiment 1."""

from __future__ import annotations

from importlib.metadata import version as _pkg_version

from langdetect import DetectorFactory, LangDetectException, detect_langs

from language_id.languages.codes import to_bcp47
from language_id.models.base import LIDPrediction


class LangdetectModel:
    """langdetect-based LID model implementing the `LIDModel` protocol.

    Note: langdetect emits "zh-cn" / "zh-tw" for Chinese variants. Whether those
    get normalized to "zh-CN"/"zh-TW" or "zh-Hans"/"zh-Hant" depends on entries
    in `languages/_mapping_overrides.yaml`. Adjust to match the gold-label
    convention of whichever benchmark you're evaluating on.
    """

    name = "langdetect"
    version = _pkg_version("langdetect")

    def __init__(self, deterministic_seed: int | None = 0) -> None:
        self.deterministic_seed = deterministic_seed
        if deterministic_seed is not None:
            DetectorFactory.seed = deterministic_seed

    def predict(self, text: str) -> LIDPrediction:
        try:
            results = detect_langs(text)
        except LangDetectException as e:
            return LIDPrediction(lang_code="und", confidence=None, raw_output=str(e))
        if not results:
            return LIDPrediction(lang_code="und", confidence=None, raw_output="")
        top = results[0]
        return LIDPrediction(
            lang_code=to_bcp47(top.lang),
            confidence=float(top.prob),
            raw_output=str(results),
        )

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in texts]
