"""Procedural memory: learned behavioral patterns and skills."""

from __future__ import annotations

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus

from .base import store_memory

# Procedural decay rate: essentially permanent
PROCEDURAL_DECAY_RATE = 0.000001


def store_procedural(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    content: str,
    trigger_context: str | None = None,
    steps: list[str] | None = None,
    tags: list[str] | None = None,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store a procedural memory (learned behavior pattern)."""
    metadata: dict = {}
    if trigger_context:
        metadata["trigger_context"] = trigger_context
    if steps:
        metadata["steps"] = steps

    return store_memory(
        session,
        embedding_service,
        content=content,
        memory_type="procedural",
        tags=tags,
        metadata=metadata,
        decay_rate=PROCEDURAL_DECAY_RATE,
        event_bus=event_bus,
    )
