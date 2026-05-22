"""Multinomial Naive Bayes over char n-gram features (spec §3 Experiment 2)."""

from __future__ import annotations

from sklearn.naive_bayes import MultinomialNB

from language_id.models.trained._sklearn_base import SklearnPipelineLIDModel


class NGramNBModel(SklearnPipelineLIDModel):
    """TfidfVectorizer(char_wb, 2-5) -> MultinomialNB."""

    name = "ngram_nb"
    version = "0.0.1"
    classifier_cls = MultinomialNB
