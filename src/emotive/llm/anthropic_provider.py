"""Anthropic LLM adapter — cloud API via official SDK.

Optional provider — requires `anthropic` package and API key.
Used for highest quality responses when local models aren't sufficient.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from emotive.config.schema import LLMProviderConfig
from emotive.logging import get_logger

from .adapter import LLMAdapter

logger = get_logger("llm.anthropic")


class AnthropicAdapter(LLMAdapter):
    """Anthropic LLM adapter using the official SDK."""

    def __init__(self, config: LLMProviderConfig) -> None:
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the 'anthropic' package. "
                "Install with: pip install 'emotive[anthropic]'"
            ) from None

        if not config.api_key:
            raise ValueError("Anthropic provider requires an API key")

        self._client = anthropic.AsyncAnthropic(api_key=config.api_key)
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens

    async def generate(
        self,
        system: str,
        messages: list[dict],
    ) -> str:
        """Non-streaming generation via Anthropic API."""
        resp = await self._client.messages.create(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return resp.content[0].text

    async def stream(
        self,
        system: str,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Streaming generation via Anthropic API."""
        async with self._client.messages.stream(
            model=self._model,
            system=system,
            messages=messages,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        ) as stream:
            async for text in stream.text_stream:
                yield text
