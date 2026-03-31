"""Abstract LLM adapter with factory for provider selection.

Single interface, multiple backends. The cognitive pipeline doesn't care
which model runs underneath — same brain, different voice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from emotive.config.schema import LLMProviderConfig


class LLMAdapter(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        system: str,
        messages: list[dict],
    ) -> str:
        """Non-streaming generation. Returns complete response text."""

    @abstractmethod
    async def stream(
        self,
        system: str,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Streaming generation. Yields text chunks."""
        ...  # pragma: no cover


def create_adapter(config: LLMProviderConfig) -> LLMAdapter:
    """Factory: create the appropriate adapter from config."""
    if config.provider == "ollama":
        from .ollama import OllamaAdapter

        return OllamaAdapter(config)
    elif config.provider == "anthropic":
        from .anthropic_provider import AnthropicAdapter

        return AnthropicAdapter(config)
    else:
        raise ValueError(f"Unknown LLM provider: {config.provider}")
