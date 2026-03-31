"""Data types for LLM adapter layer."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class LLMMessage:
    """A single message in the conversation."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class LLMResponse:
    """Complete (non-streaming) response from the LLM."""

    content: str
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A single chunk from a streaming LLM response."""

    text: str
    done: bool = False
