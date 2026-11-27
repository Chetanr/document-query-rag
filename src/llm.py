from __future__ import annotations

import logging

from anthropic import Anthropic

logger = logging.getLogger(__name__)


class LLMConfigurationError(RuntimeError):
    pass


class LLM:
    def generate(self, system: str, user: str) -> str:
        raise NotImplementedError


class AnthropicLLM(LLM):
    def __init__(self, model: str | None, api_key: str | None):
        self.model = model
        self.api_key = api_key
        self._client: Anthropic | None = None

    def _get_client(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=self.api_key)
        return self._client

    def generate(self, system: str, user: str) -> str:
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not set. Cannot generate answer.")
            raise LLMConfigurationError("Set ANTHROPIC_API_KEY to use the ask endpoint")
        if not self.model:
            raise LLMConfigurationError("Set ANTHROPIC_MODEL in the .env file")

        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        answer = "\n".join(
            block.text for block in response.content if block.type == "text"
        )
        if not answer.strip():
            logger.warning("Anthropic returned an empty answer")
            raise RuntimeError("Anthropic returned an empty answer")
        return answer
