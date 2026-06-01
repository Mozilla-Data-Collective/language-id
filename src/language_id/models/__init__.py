"""Model registry. `get_model(name)` returns a ready-to-use LID model.

LLMs are served via the Together SDK (add entries to `TOGETHER_MODELS`).
Standard tools are langdetect / GlotLID / NLLB-LID.
"""

from language_id.models.base import LIDModel, LIDPrediction

# Short name -> Together model ID
TOGETHER_MODELS = {
    "qwen": "Qwen/Qwen3.7-Max",
    "gemma": "google/gemma-4-31B-it",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "deepseek": "deepseek-ai/deepseek-v4-pro",
    "kimi": "moonshotai/kimi-k2.7",
}

STANDARD_MODELS = ("langdetect", "glotlid", "nllb-lid")


def get_model(name: str) -> LIDModel:
    """Instantiate a model by short name."""
    if name in TOGETHER_MODELS:
        from language_id.models.together import TogetherModel

        return TogetherModel(model_id=TOGETHER_MODELS[name], name=name)
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
