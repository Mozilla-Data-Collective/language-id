from typing import Any

from language_id.models.llm.base import BaseLLMModel


class TogetherModel(BaseLLMModel):
    """LID model wrapping `together.Together().chat.completions.create`.

    Few-shot examples (if any) are injected as alternating user/assistant turns
    between the system prompt and the final user message.
    """

    def __init__(
        self,
        name: str,
        version: str,
        provider: str,
        model_id: str,
        prompt: dict[str, str],
        client_options: dict[str, Any] | None = None,
        few_shot_examples: list[tuple[str, str]] | None = None,
    ) -> None:
        opts = client_options or {}
        examples = list(few_shot_examples or [])
        super().__init__(
            name=name,
            version=version,
            max_retries=int(opts.get("max_retries", 3)),
        )
        self.model_id = model_id
        self.system_prompt = prompt["system"]
        self.user_template = prompt["user_template"]
        self.client_options = opts
        self.few_shot_examples = examples
        self._client: Any | None = None  # lazy; reads TOGETHER_API_KEY from env

    def _get_client(self) -> Any:
        if self._client is None:
            from together import Together

            self._client = Together()
        return self._client

    def _raw_call(self, text: str) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        for ex_text, ex_lang in self.few_shot_examples:
            messages.append(
                {"role": "user", "content": self.user_template.format(text=ex_text)}
            )
            messages.append({"role": "assistant", "content": ex_lang})
        messages.append(
            {"role": "user", "content": self.user_template.format(text=text)}
        )

        client = self._get_client()
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            messages=messages,
            temperature=self.client_options.get("temperature", 0.0),
            max_tokens=self.client_options.get("max_tokens", 64),
            timeout=self.client_options.get("timeout_s", 60),
        )

        if self.client_options.get("stream", False):
            parts: list[str] = []
            for chunk in client.chat.completions.create(stream=True, **kwargs):
                delta = chunk.choices[0].delta.content
                if delta:
                    parts.append(delta)
            return "".join(parts)

        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""