"""Bounded working memory buffer. In-memory only, never persisted to DB."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone

from pydantic import BaseModel, Field


class WorkingMemoryItem(BaseModel):
    """A single item in working memory."""

    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    significance: float = Field(default=0.5, ge=0.0, le=1.0)
    conversation_id: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class WorkingMemory:
    """Bounded working memory buffer with LRU eviction.

    Items evicted at capacity are published to the event bus
    so consolidation can decide whether to promote them.
    """

    def __init__(self, capacity: int = 20, event_bus: object | None = None) -> None:
        self._capacity = capacity
        self._buffer: deque[WorkingMemoryItem] = deque()
        self._event_bus = event_bus

    def add(self, item: WorkingMemoryItem) -> WorkingMemoryItem | None:
        """Add an item. Returns the evicted item if buffer was at capacity, else None."""
        evicted = None
        if len(self._buffer) >= self._capacity:
            evicted = self._buffer.popleft()
            if self._event_bus is not None:
                from emotive.runtime.event_bus import WORKING_MEMORY_EVICTED

                self._event_bus.publish(
                    WORKING_MEMORY_EVICTED,
                    {
                        "content": evicted.content,
                        "significance": evicted.significance,
                        "timestamp": evicted.timestamp.isoformat(),
                    },
                )
        self._buffer.append(item)
        return evicted

    def get_all(self) -> list[WorkingMemoryItem]:
        """Return all items in working memory."""
        return list(self._buffer)

    def get_above_threshold(self, significance: float) -> list[WorkingMemoryItem]:
        """Return items with significance >= threshold."""
        return [item for item in self._buffer if item.significance >= significance]

    def clear(self) -> list[WorkingMemoryItem]:
        """Clear working memory. Returns all items that were in it."""
        items = list(self._buffer)
        self._buffer.clear()
        return items

    @property
    def size(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity
