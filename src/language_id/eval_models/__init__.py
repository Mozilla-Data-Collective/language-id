"""Eval-model registry. `get_eval_model(name)` returns a ready-to-use LID model.

"Eval models" are the off-the-shelf models you *evaluate* with `language-id eval`
(standard tools + LLMs) — as opposed to the "train models" in `language_id.train`
(naive Bayes / logistic regression / fine-tuned HF models) that you *train* with
`language-id train`.

LLMs are served via the Together SDK (add entries to `TOGETHER_MODELS`).
Standard tools are langdetect / GlotLID / NLLB-LID.
"""

from language_id.eval_models.base import LIDModel, LIDPrediction
from language_id.eval_models.together import TOGETHER_MODELS

STANDARD_MODELS = ("langdetect", "glotlid", "nllb-lid")


def get_eval_model(name: str, examples: list[tuple[str, str]] | None = None) -> LIDModel:
    """Instantiate an eval model by short name.

    `examples` are (text, iso639-3) pairs for few-shot prompting. Only LLMs
    use them (standard tools ignore them).
    """
    if name in TOGETHER_MODELS:
        from language_id.eval_models.together import TogetherModel

        return TogetherModel(model_id=TOGETHER_MODELS[name], name=name, examples=examples)
    if examples:
        raise ValueError(f"few-shot examples are only supported for LLMs, not {name!r}.")
    if name == "langdetect":
        from language_id.eval_models.langdetect import LangdetectModel

        return LangdetectModel()
    if name == "glotlid":
        from language_id.eval_models.fasttext import GlotLIDModel

        return GlotLIDModel()
    if name == "nllb-lid":
        from language_id.eval_models.fasttext import NLLBLIDModel

        return NLLBLIDModel()
    available = ", ".join([*TOGETHER_MODELS, *STANDARD_MODELS])
    raise ValueError(f"unknown eval model {name!r}. Available: {available}")


def available_eval_models() -> list[str]:
    return [*TOGETHER_MODELS, *STANDARD_MODELS]


__all__ = [
    "STANDARD_MODELS",
    "TOGETHER_MODELS",
    "LIDModel",
    "LIDPrediction",
    "available_eval_models",
    "get_eval_model",
]
