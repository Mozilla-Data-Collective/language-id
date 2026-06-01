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
        max_tokens: int = 16,
        timeout_s: int = 120,
        max_retries: int = 3,
        stream: bool = True,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.stream = stream
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from together import Together

            self._client = Together()
        return self._client

    def _raw_call(self, text: str) -> str:
        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(text=text)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=self.timeout_s,
        )
        if self.stream:
            parts = [
                chunk.choices[0].delta.content or ""
                for chunk in client.chat.completions.create(stream=True, **kwargs)
            ]
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
