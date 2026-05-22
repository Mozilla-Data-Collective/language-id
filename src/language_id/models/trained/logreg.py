"""Logistic Regression over char n-gram TF-IDF features (spec §3 Experiment 2)."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from language_id.models.trained._sklearn_base import SklearnPipelineLIDModel


class LogRegModel(SklearnPipelineLIDModel):
    """TfidfVectorizer(char_wb, 2-5) -> LogisticRegression."""

    name = "logreg"
    version = "0.0.1"
    classifier_cls = LogisticRegression
