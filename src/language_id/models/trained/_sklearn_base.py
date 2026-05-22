"""Shared base for sklearn-pipeline LID models (LogReg, MultinomialNB).

Subclasses define `classifier_cls`. The vectorizer is always TfidfVectorizer
configured from YAML (spec §3 Experiment 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

from language_id.models.base import LIDPrediction


class SklearnPipelineLIDModel:
    """Base for sklearn-based LID models implementing the `LIDModel` protocol."""

    name: ClassVar[str] = "sklearn"
    version: ClassVar[str] = "0.0.1"
    classifier_cls: ClassVar[type | None] = None

    def __init__(
        self,
        vectorizer: dict[str, Any] | None = None,
        classifier: dict[str, Any] | None = None,
    ) -> None:
        self.vectorizer_cfg = dict(vectorizer or {})
        self.classifier_cfg = dict(classifier or {})
        self.pipeline: Pipeline | None = None

    def _build_pipeline(self) -> Pipeline:
        if self.classifier_cls is None:
            raise NotImplementedError(f"{type(self).__name__} must set classifier_cls")
        vec_cfg = dict(self.vectorizer_cfg)
        if "ngram_range" in vec_cfg:
            vec_cfg["ngram_range"] = tuple(vec_cfg["ngram_range"])
        return Pipeline(
            [
                ("tfidf", TfidfVectorizer(**vec_cfg)),
                ("clf", self.classifier_cls(**self.classifier_cfg)),
            ]
        )

    def fit(self, df: pd.DataFrame, text_col: str, lang_col: str) -> None:
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df[text_col].tolist(), df[lang_col].tolist())

    def predict(self, text: str) -> LIDPrediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} not fit yet; call fit() or load()")
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            probs = self.pipeline.predict_proba(texts)
            classes = clf.classes_
            return [
                LIDPrediction(
                    lang_code=str(classes[int(np.argmax(p))]),
                    confidence=float(np.max(p)),
                    raw_output="",
                )
                for p in probs
            ]
        preds = self.pipeline.predict(texts)
        return [
            LIDPrediction(lang_code=str(p), confidence=None, raw_output="")
            for p in preds
        ]

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        if self.pipeline is None:
            raise RuntimeError(f"{type(self).__name__} not fit yet; call fit() or load()")
        probs = self.pipeline.predict_proba(texts)
        classes = self.pipeline.named_steps["clf"].classes_
        return [
            {str(c): float(p) for c, p in zip(classes, prob, strict=True)} for prob in probs
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "vectorizer_cfg": self.vectorizer_cfg,
                "classifier_cfg": self.classifier_cfg,
                "name": self.name,
                "version": self.version,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> "SklearnPipelineLIDModel":
        path = Path(path)
        if path.is_dir():
            path = path / "model.joblib"
        data = joblib.load(path)
        instance = cls(
            vectorizer=data.get("vectorizer_cfg") or {},
            classifier=data.get("classifier_cfg") or {},
        )
        instance.pipeline = data["pipeline"]
        return instance
