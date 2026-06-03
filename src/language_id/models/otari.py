import os
import time
from typing import Any

from dotenv import load_dotenv
from otari import OtariClient

from language_id.models.base import LIDPrediction
from language_id.models.together import SYSTEM_PROMPT, USER_TEMPLATE, _parse

"""
NOTE: Right now this client is not used for LLM inference. Its placed here for future
migration as otari allows for LLM inference from both local models (e.g. ollama) and 
a broad list of different providers (TogetherAI, OpenAI, Anthropic, Mistral, etc).

If you do not have a TogetherAI API key, you can simply start with a local LLM using Ollama
through the Otari client.
"""


PROVIDER_PREFIX = "together:"  # Otari routes these ids to the Together provider

class OtariModel:
    """A chat LLM used as an LID classifier, served via the Otari gateway.
    Can be used as a drop-in replacement for a TogetherAI client.

    Authenticates with `OTARI_PLATFORM_TOKEN`. Few-shot `examples` are
    (text, iso639-3) pairs prepended to the prompt as user/assistant turns.
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
        examples: list[tuple[str, str]] | None = None,
    ) -> None:
        self.model_id = model_id
        self.name = name or model_id
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.examples = examples or []
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:

            load_dotenv()
            token = os.getenv("OTARI_PLATFORM_TOKEN")
            if not token:
                raise RuntimeError("Missing OTARI_PLATFORM_TOKEN environment variable")
            self._client = OtariClient(platform_token=token)
        return self._client

    def _messages(self, text: str) -> list[dict[str, str]]:
        """System prompt, then few-shot demos as turns, then the text to classify."""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for ex_text, ex_code in self.examples:
            messages.append({"role": "user", "content": USER_TEMPLATE.format(text=ex_text)})
            messages.append({"role": "assistant", "content": ex_code})
        messages.append({"role": "user", "content": USER_TEMPLATE.format(text=text)})
        return messages

    def _call(self, text: str) -> str:
        response = self._get_client().completion(
            model=f"{PROVIDER_PREFIX}{self.model_id}",
            messages=self._messages(text),
            temperature=self.temperature,
            max_tokens=self.max_output_tokens,
            timeout=self.timeout_s,
        )
        return response.choices[0].message.content or ""

    def predict(self, text: str) -> LIDPrediction:
        for attempt in range(self.max_retries):
            try:
                raw = self._call(text)
                break
            except Exception:
                if attempt + 1 == self.max_retries:
                    raise
                time.sleep(2**attempt)  # back off, then retry
        return LIDPrediction(lang_code=_parse(raw), confidence=None, raw_output=raw)

    def predict_batch(self, texts: list[str]) -> list[LIDPrediction]:
        return [self.predict(t) for t in texts]
