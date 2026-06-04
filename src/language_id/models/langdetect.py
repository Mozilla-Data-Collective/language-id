from langdetect import DetectorFactory, LangDetectException, detect_langs
from tqdm import tqdm

from language_id.lang_codes_mapping import to_iso3
from language_id.models.base import LIDPrediction


class LangdetectModel:
    """langdetect-based LID model. Emits ISO-639-1 internally and it gets normalized to ISO-639-3."""

    name = "langdetect"

    def __init__(self, deterministic_seed: int | None = 0) -> None:
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
            lang_code=to_iso3(top.lang),
            confidence=float(top.prob),
            raw_output=str(results),
        )

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in tqdm(texts, desc=self.name, unit="text")]
