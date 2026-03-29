"""Semantic memory: generalized patterns extracted from episodic clusters."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import create_memory_link
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import EventBus

from .base import store_memory

# Semantic decay rate: half-life ~10,000 days
SEMANTIC_DECAY_RATE = 0.00001


def store_semantic(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    content: str,
    source_memory_ids: list[uuid.UUID] | None = None,
    confidence: float = 0.5,
    tags: list[str] | None = None,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store a semantic memory (generalized knowledge from patterns)."""
    metadata: dict = {}
    if source_memory_ids:
        metadata["origin_memories"] = [str(mid) for mid in source_memory_ids]

    mem = store_memory(
        session,
        embedding_service,
        content=content,
        memory_type="semantic",
        tags=tags,
        metadata=metadata,
        decay_rate=SEMANTIC_DECAY_RATE,
        confidence=confidence,
        event_bus=event_bus,
    )

    # Link back to origin episodic memories
    if source_memory_ids:
        for src_id in source_memory_ids:
            create_memory_link(
                session, src_id, mem.id, "conceptual_overlap", strength=0.7
            )

    return mem


def extract_semantic_from_cluster(
    session: Session,
    embedding_service: EmbeddingService,
    cluster_memories: list[Memory],
    *,
    event_bus: EventBus | None = None,
) -> Memory | None:
    """Given a cluster of similar episodic memories, extract a semantic pattern.

    Generates a summary statement from the cluster contents and stores it
    as a semantic memory linked back to all source episodics.
    """
    if len(cluster_memories) < 2:
        return None

    # Build a summary by combining the cluster contents
    # In Phase 0, this is a simple concatenation-based summary
    # Future phases could use an LLM for better extraction
    contents = [m.content for m in cluster_memories]
    common_tags = _find_common_tags(cluster_memories)

    # Create a pattern statement from the cluster
    pattern = f"Pattern from {len(cluster_memories)} observations: " + " | ".join(
        c[:100] for c in contents
    )

    source_ids = [m.id for m in cluster_memories]
    confidence = min(len(cluster_memories) / 10.0, 1.0)

    return store_semantic(
        session,
        embedding_service,
        content=pattern,
        source_memory_ids=source_ids,
        confidence=confidence,
        tags=common_tags,
        event_bus=event_bus,
    )


def _find_common_tags(memories: list[Memory]) -> list[str]:
    """Find tags that appear in at least half the memories."""
    if not memories:
        return []
    tag_counts: dict[str, int] = {}
    for m in memories:
        for tag in (m.tags or []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    threshold = len(memories) / 2
    return [tag for tag, count in tag_counts.items() if count >= threshold]
