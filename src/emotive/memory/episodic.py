"""Episodic memory: specific events with rich context."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus

from .base import store_memory

# Episodic decay rate: half-life ~1000 days
EPISODIC_DECAY_RATE = 0.0001


def store_episodic(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    content: str,
    conversation_id: uuid.UUID | None = None,
    tags: list[str] | None = None,
    context: dict | None = None,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store an episodic memory (specific event with context)."""
    metadata = {}
    if context:
        metadata.update(context)

    return store_memory(
        session,
        embedding_service,
        content=content,
        memory_type="episodic",
        conversation_id=conversation_id,
        tags=tags,
        metadata=metadata,
        decay_rate=EPISODIC_DECAY_RATE,
        event_bus=event_bus,
    )
