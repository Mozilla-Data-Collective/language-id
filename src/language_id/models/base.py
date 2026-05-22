"""The `LIDModel` protocol (spec §6).

Every model — frontier LLM, classical baseline, trained classifier — implements
this interface. The evaluation loop is identical regardless of model kind.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class LIDPrediction:
    lang_code: str                  # BCP-47, e.g. "en", "zh-Hans", "kab"
    confidence: float | None        # 0.0-1.0 if available, else None
    raw_output: str                 # for audit, especially for LLMs


@runtime_checkable
class LIDModel(Protocol):
    name: str
    version: str

    def predict(self, text: str) -> LIDPrediction: ...

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]: ...
