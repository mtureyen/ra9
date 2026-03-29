"""Sensory buffer: raw input staging before processing.

In Phase 0 this is pass-through (no appraisal filter).
Future phases will add emotional appraisal here.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel


class ProcessedInput(BaseModel):
    """Result of sensory buffer processing."""

    text: str
    timestamp: datetime
    char_count: int
    truncated: bool = False


# mxbai-embed-large has a 512 token context window (~2000 chars conservative estimate)
MAX_INPUT_CHARS = 2000


class SensoryBuffer:
    """Receives raw input, performs minimal preprocessing.

    Phase 0: pass-through with truncation.
    Phase 1+: will add appraisal filter here.
    """

    def __init__(self, max_chars: int = MAX_INPUT_CHARS) -> None:
        self._max_chars = max_chars

    def process(self, raw_input: str) -> ProcessedInput:
        """Process raw input text through the sensory buffer."""
        text = raw_input.strip()
        truncated = len(text) > self._max_chars
        if truncated:
            text = text[: self._max_chars]

        return ProcessedInput(
            text=text,
            timestamp=datetime.now(timezone.utc),
            char_count=len(text),
            truncated=truncated,
        )
