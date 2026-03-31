"""Conversation buffer with gist compression.

Manages the active conversation window for LLM context. When turns exit
the active buffer, they're compressed into gist summaries.

Brain analog: dlPFC working memory (~4 chunks) + hippocampal gist extraction.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationBuffer:
    """Bounded conversation buffer with primacy pinning.

    Keeps the most recent turns in the active window. First N turns
    are pinned (primacy anchoring) and never evicted. When non-pinned
    turns exceed the buffer size, oldest are evicted for gist compression.

    # TODO: Phase 2+ — relevance-scored retention (recency + emotional
    # valence + goal-alignment) and topic-change suppression per Phase 1.5 doc.
    """

    def __init__(
        self,
        buffer_size: int = 6,
        primacy_pins: int = 2,
    ) -> None:
        self._buffer_size = buffer_size
        self._primacy_pins = primacy_pins
        self._pinned: list[ConversationTurn] = []
        self._active: deque[ConversationTurn] = deque()
        self._full_session: list[ConversationTurn] = []

    def add_turn(self, role: str, content: str) -> list[ConversationTurn]:
        """Add a turn to the buffer.

        Returns any turns that were evicted from the active window
        (these should be compressed into gist summaries).
        """
        turn = ConversationTurn(role=role, content=content)
        self._full_session.append(turn)

        # Pin early turns
        if len(self._pinned) < self._primacy_pins:
            self._pinned.append(turn)
            return []

        self._active.append(turn)

        # Evict oldest non-pinned turns if over capacity
        max_active = self._buffer_size - len(self._pinned)
        evicted = []
        while len(self._active) > max_active:
            evicted.append(self._active.popleft())

        return evicted

    def get_active_turns(self) -> list[ConversationTurn]:
        """Get all turns in the active window (pinned + recent)."""
        return self._pinned + list(self._active)

    def get_full_session(self) -> list[ConversationTurn]:
        """Get all turns from the entire session."""
        return list(self._full_session)

    def clear(self) -> None:
        """Clear the buffer (for new sessions)."""
        self._pinned.clear()
        self._active.clear()
        self._full_session.clear()


def compress_to_gist(turns: list[ConversationTurn]) -> str:
    """Extract a gist summary from evicted conversation turns.

    Currently extractive (concatenation + truncation). No LLM call.
    The gist is stored as an episodic memory by the hippocampus.

    Future: LLM-based abstractive summarization as a background task.
    """
    parts = []
    for turn in turns:
        # Truncate long turns to keep gist compact
        content = turn.content[:300]
        if len(turn.content) > 300:
            content += "..."
        parts.append(f"{turn.role}: {content}")

    return "Conversation summary: " + " | ".join(parts)
