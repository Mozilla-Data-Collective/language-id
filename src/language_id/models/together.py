import time
from typing import Any

from language_id.lang_codes_mapping import to_iso3
from language_id.models.base import LIDPrediction

SYSTEM_PROMPT = (
    "You are an expert language identification system. Given a piece of text (short or long), "
    "identify its language and respond with ONLY the ISO 639-3 code (three lowercase "
    'letters, e.g. "eng", "kab", "arb"). No explanations, no extra words.'
)
USER_TEMPLATE = "Text:\n{text}\n\nISO 639-3 code:"

# Short name -> Together model ID
TOGETHER_MODELS = {
    # "deepseek": "deepseek-ai/deepseek-v4-pro",  #1.6T parameters (49B activated) - reasoning
    # "minimax-m27": "MiniMaxAI/MiniMax-M2.7",  # 230b, 10b activated
    "qwen": "Qwen/Qwen3.7-Max",  # Closed source / 1T - reasoning
    "gpt-oss-120b": "openai/gpt-oss-120b",  #120b
    "gemma": "google/gemma-4-31B-it",  #32b
    "gpt-oss-20b": "openai/gpt-oss-20b", #20b
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",  # 8b
}

class TogetherModel:
    """Together-hosted chat model used as an LID classifier.

    Reads `TOGETHER_API_KEY` from the environment (via the `together` SDK).
    Some snapshots only support streaming, so `stream=True` accumulates chunks.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        temperature: float = 0.0,
        # Reasoning models (e.g. gpt-oss) spend tokens on hidden reasoning before
        # emitting the answer; a small cap leaves no room for the code itself.
        # Non-reasoning models stop right after the short code, so this is a
        # ceiling, not a cost.
        max_tokens: int = 256,
        timeout_s: int = 120,
        max_retries: int = 3,
        stream: bool = True,
        examples: list[tuple[str, str]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.stream = stream
        # (text, iso639-3) demonstration pairs for few-shot prompting.
        self.examples = examples or []
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from together import Together

            self._client = Together()
        return self._client

    def _raw_call(self, text: str) -> str:
        client = self._get_client()
        # System prompt, then any few-shot demonstrations as user/assistant turns,
        # then the text to classify.
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_text, ex_code in self.examples:
            messages.append({"role": "user", "content": USER_TEMPLATE.format(text=ex_text)})
            messages.append({"role": "assistant", "content": ex_code})
        messages.append({"role": "user", "content": USER_TEMPLATE.format(text=text)})
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_s,
        )
        if self.stream:
            parts = []
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                # Some chunks (e.g. the final usage/keepalive chunk) carry no
                # choices; skip anything without a content delta.
                if not chunk.choices:
                    continue
                parts.append(chunk.choices[0].delta.content or "")
            return "".join(parts)
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def predict(self, text: str) -> LIDPrediction:
        raw = ""
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                raw = self._raw_call(text)
                break
            except Exception as e:
                last_exc = e
                if attempt + 1 < self.max_retries:
                    time.sleep(2**attempt)
        else:
            raise last_exc  # type: ignore[misc]
        return LIDPrediction(lang_code=_parse(raw), confidence=None, raw_output=raw)

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in texts]


def _parse(raw: str) -> str:
    """Best-effort ISO-639-3 from a model response.

    Tries the whole cleaned string (handles a bare code or a language name like
    "Standard Arabic"), then falls back to the first whitespace token.
    """
    text = raw.strip().strip("`").strip("\"'").strip().rstrip(".,;:!?")
    code = to_iso3(text)
    if code != "und":
        return code
    first = text.split()[0] if text.split() else ""
    return to_iso3(first)
