"""Model registry. `get_model(name)` returns a ready-to-use LID model.

LLMs are served via the Together SDK (add entries to `TOGETHER_MODELS`).
Standard tools are langdetect / GlotLID / NLLB-LID.
"""

from language_id.models.base import LIDModel, LIDPrediction
from language_id.models.together import TOGETHER_MODELS

STANDARD_MODELS = ("langdetect", "glotlid", "nllb-lid")


def get_model(name: str, examples: list[tuple[str, str]] | None = None) -> LIDModel:
    """Instantiate a model by short name.

    `examples` are (text, iso639-3) pairs for few-shot prompting. Only LLMs
    use them (standard tools ignore them).
    """
    if name in TOGETHER_MODELS:
        from language_id.models.together import TogetherModel

        return TogetherModel(model_id=TOGETHER_MODELS[name], name=name, examples=examples)
    if examples:
        raise ValueError(f"few-shot examples are only supported for LLMs, not {name!r}.")
    if name == "langdetect":
        from language_id.models.langdetect import LangdetectModel

        return LangdetectModel()
    if name == "glotlid":
        from language_id.models.fasttext import GlotLIDModel

        return GlotLIDModel()
    if name == "nllb-lid":
        from language_id.models.fasttext import NLLBLIDModel

        return NLLBLIDModel()
    available = ", ".join([*TOGETHER_MODELS, *STANDARD_MODELS])
    raise ValueError(f"unknown model {name!r}. Available: {available}")


def available_models() -> list[str]:
    return [*TOGETHER_MODELS, *STANDARD_MODELS]


__all__ = [
    "STANDARD_MODELS",
    "TOGETHER_MODELS",
    "LIDModel",
    "LIDPrediction",
    "available_models",
    "get_model",
]
