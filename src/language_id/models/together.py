import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from tqdm import tqdm

from language_id.lang_codes_mapping import to_iso3
from language_id.models.base import LIDPrediction

SYSTEM_PROMPT = (
    "You are an expert language identification system. Given a piece of text (short or long), "
    "identify its language and respond with ONLY the ISO 639-3 code (three lowercase "
    'letters, e.g. "eng", "kab", "arb"). No explanations, no extra words.'
)
USER_TEMPLATE = "Text:\n{text}\n\nISO 639-3 code:"

# Short name -> Together model ID.
TOGETHER_MODELS = {
    # "deepseek": "deepseek-ai/deepseek-v4-pro",  # 1.6T params (49B activated) - reasoning
    "minimax-m27": "MiniMaxAI/MiniMax-M2.7",  # 230b, 10b activated
    # "qwen": "Qwen/Qwen3.7-Max",  # closed source / 1T - reasoning
    "gpt-oss-120b": "openai/gpt-oss-120b",  # 120b
    "gemma": "google/gemma-4-31B-it",  # 32b
    "gpt-oss-20b": "openai/gpt-oss-20b",  # 20b
    "llama": "meta-llama/Meta-Llama-3-8B-Instruct-Lite",  # 8b
}


class TogetherModel:
    """A chat LLM used as an LID classifier, served via the Together SDK.

    Authenticates with `TOGETHER_API_KEY` (read by the Together client from the
    environment). Few-shot `examples` are (text, iso639-3) pairs prepended to the
    prompt as user/assistant turns.
    """

    def __init__(
        self,
        model_id: str,
        *,
        name: str | None = None,
        temperature: float = 0.0,
        # Output-token ceiling. Reasoning models (e.g. gpt-oss) also spend tokens
        # here on hidden reasoning, so too small a cap leaves no room for the code.
        max_output_tokens: int = 512,
        timeout_s: int = 120,
        max_retries: int = 3,
        # Calls are I/O-bound HTTP requests, so run a batch concurrently. Bump
        # this up for more speed, down if you hit rate limits (set 1 to disable).
        max_workers: int = 8,
        examples: list[tuple[str, str]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.max_workers = max_workers
        self.examples = examples or []
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from together import Together

            self._client = Together()
        return self._client

    def _messages(self, text: str) -> list[dict[str, str]]:
        """System prompt, then few-shot demos as turns, then the text to classify."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_text, ex_code in self.examples:
            messages.append({"role": "user", "content": USER_TEMPLATE.format(text=ex_text)})
            messages.append({"role": "assistant", "content": ex_code})
        messages.append({"role": "user", "content": USER_TEMPLATE.format(text=text)})
        return messages

    def _raw_call(self, text: str) -> str:
        completion = self._get_client().chat.completions.create(
            model=self.model_id,
            messages=self._messages(text),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            timeout=self.timeout_s,
        )
        return completion.choices[0].message.content or ""

    def predict(self, text: str) -> LIDPrediction:
        for attempt in range(self.max_retries):
            try:
                raw = self._raw_call(text)
                break
            except Exception:
                if attempt + 1 == self.max_retries:
                    raise
                time.sleep(2**attempt)  # back off, then retry
        return LIDPrediction(lang_code=_parse(raw), confidence=None, raw_output=raw)

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        if self.max_workers <= 1 or len(texts) <= 1:
            return [self.predict(t) for t in tqdm(texts, desc=self.name, unit="text")]
        # One shared client, created up front so the workers don't race to init it.
        self._get_client()
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            # map preserves input order
            return list(tqdm(pool.map(self.predict, texts), total=len(texts), desc=self.name, unit="text"))


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
