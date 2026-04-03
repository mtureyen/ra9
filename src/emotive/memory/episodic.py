"""Episodic memory: specific events with rich context."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from emotive.db.models.episode import EmotionalEpisode
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
    decay_rate: float | None = None,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store an episodic memory (specific event with context).

    decay_rate: override the default episodic decay rate. Lower values
    decay slower (more significant memories persist longer).
    """
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
        decay_rate=decay_rate if decay_rate is not None else EPISODIC_DECAY_RATE,
        event_bus=event_bus,
    )


def store_episodic_from_episode(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    episode: EmotionalEpisode,
    content: str,
    conversation_id: uuid.UUID | None = None,
    tags: list[str] | None = None,
    encoding_strength_weight: float = 0.8,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store an episodic memory from an emotional episode.

    Emotional intensity determines encoding strength:
    encoding_strength = intensity * weight + (1 - weight)
    Higher intensity = stronger encoding = more decay protection.
    """
    encoding_strength = episode.intensity * encoding_strength_weight + (
        1 - encoding_strength_weight
    )
    # More protection for stronger encoding (lower = slower decay)
    decay_protection = 1.0 - (encoding_strength * 0.5)

    # Flashbulb memories: extremely high intensity formative events
    # get near-zero decay — like flashbulb memories in humans that
    # persist for years with vivid detail. The gaslighting attack,
    # the naming, the first time someone said "I care about you."
    if episode.is_formative and episode.intensity > 0.8:
        decay_protection = 0.1  # 10% of normal decay — near permanent

    mem = store_memory(
        session,
        embedding_service,
        content=content,
        memory_type="episodic",
        conversation_id=conversation_id,
        tags=tags,
        decay_rate=EPISODIC_DECAY_RATE,
        emotional_intensity=episode.intensity,
        primary_emotion=episode.primary_emotion,
        valence=episode.appraisal_valence,
        source_episode_id=episode.id,
        decay_protection=decay_protection,
        event_bus=event_bus,
    )

    # Mark episode as encoded
    episode.memory_encoded = True
    session.flush()

    return mem
