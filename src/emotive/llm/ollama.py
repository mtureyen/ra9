"""Ollama LLM adapter — local inference via HTTP API.

Supports both localhost and remote Ollama instances (MacBook+PC deployment).
Uses httpx for async HTTP with streaming support.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from emotive.config.schema import LLMProviderConfig
from emotive.logging import get_logger

from .adapter import LLMAdapter

logger = get_logger("llm.ollama")


class OllamaAdapter(LLMAdapter):
    """Ollama LLM adapter using the /api/chat endpoint."""

    def __init__(self, config: LLMProviderConfig) -> None:
        self._host = config.host.rstrip("/")
        self._model = config.model
        self._temperature = config.temperature
        self._max_tokens = config.max_tokens
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    async def generate(
        self,
        system: str,
        messages: list[dict],
    ) -> str:
        """Non-streaming generation via Ollama."""
        payload = self._build_payload(system, messages, stream=False)
        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    async def stream(
        self,
        system: str,
        messages: list[dict],
    ) -> AsyncIterator[str]:
        """Streaming generation via Ollama. Yields text chunks."""
        payload = self._build_payload(system, messages, stream=True)
        async with self._client.stream(
            "POST", "/api/chat", json=payload
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                content = data.get("message", {}).get("content", "")
                if content:
                    yield content
                if data.get("done", False):
                    break

    def _build_payload(
        self,
        system: str,
        messages: list[dict],
        *,
        stream: bool,
    ) -> dict:
        """Build the Ollama API request payload."""
        all_messages = [{"role": "system", "content": system}] + messages
        return {
            "model": self._model,
            "messages": all_messages,
            "stream": stream,
            "options": {
                "temperature": self._temperature,
                "num_predict": self._max_tokens,
                "stop": ["You:", "User:", "\nYou:", "\nUser:"],
            },
        }

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
