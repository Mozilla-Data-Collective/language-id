from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LIDPrediction:
    lang_code: str                  # ISO-639-3, e.g. "eng", "kab" (or "und")
    confidence: float | None        # 0.0-1.0 if available, else None
    raw_output: str                 # for audit, especially for LLMs


@runtime_checkable
class LIDModel(Protocol):
    """
    Every model (LLM or standard tool) implements this interface, so the
    evaluation loop is identical regardless of model kind.
    """

    name: str

    def predict(self, text: str) -> LIDPrediction: ...

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]: ...
