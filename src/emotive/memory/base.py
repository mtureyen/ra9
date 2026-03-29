"""Shared memory operations: store and recall across all memory types."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import (
    get_linked_memories,
    rank_memories,
    search_by_embedding,
)
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import MEMORY_RECALLED, MEMORY_STORED, EventBus


def store_memory(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    content: str,
    memory_type: str,
    conversation_id: uuid.UUID | None = None,
    tags: list[str] | None = None,
    metadata: dict | None = None,
    decay_rate: float | None = None,
    confidence: float = 1.0,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store a memory with auto-generated embedding."""
    embedding = embedding_service.embed_text(content)

    mem = Memory(
        memory_type=memory_type,
        content=content,
        embedding=embedding,
        conversation_id=conversation_id,
        tags=tags or [],
        confidence=confidence,
    )
    if metadata:
        mem.metadata_ = metadata
    if decay_rate is not None:
        mem.decay_rate = decay_rate

    session.add(mem)
    session.flush()

    if event_bus:
        event_bus.publish(
            MEMORY_STORED,
            {"memory_type": memory_type, "content": content[:200]},
            memory_id=mem.id,
            conversation_id=conversation_id,
        )

    return mem


def recall_memories(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    query: str,
    memory_type: str | None = None,
    limit: int = 5,
    include_spreading: bool = True,
    w_semantic: float = 0.5,
    w_recency: float = 0.3,
    w_activation: float = 0.2,
    spreading_hops: int = 2,
    spreading_decay: float = 0.6,
    event_bus: EventBus | None = None,
    conversation_id: uuid.UUID | None = None,
) -> list[dict]:
    """Full retrieval pipeline: embed → search → spread → rank → mark labile."""
    query_embedding = embedding_service.embed_text(query)

    # Step 1: Vector search (fetch more candidates than limit for spreading)
    candidates = search_by_embedding(
        session,
        query_embedding,
        limit=limit * 4,
        memory_type=memory_type,
    )

    if not candidates:
        return []

    # Step 2: Spreading activation
    activation_scores: dict[uuid.UUID, float] = {}
    if include_spreading and candidates:
        seed_ids = [c["id"] for c in candidates[:limit]]
        activation_scores = get_linked_memories(
            session,
            seed_ids,
            hops=spreading_hops,
            decay_per_hop=spreading_decay,
        )

    # Step 3: Rank
    ranked = rank_memories(
        candidates,
        activation_scores,
        w_semantic=w_semantic,
        w_recency=w_recency,
        w_activation=w_activation,
    )

    # Step 4: Take top results, mark as labile, update retrieval stats
    results = ranked[:limit]
    now = datetime.now(timezone.utc)
    from datetime import timedelta

    labile_until = now + timedelta(hours=1)

    for r in results:
        mid = r["id"]
        session.execute(
            Memory.__table__.update()
            .where(Memory.__table__.c.id == mid)
            .values(
                retrieval_count=Memory.__table__.c.retrieval_count + 1,
                last_retrieved=now,
                is_labile=True,
                labile_until=labile_until,
            )
        )

        if event_bus:
            event_bus.publish(
                MEMORY_RECALLED,
                {
                    "content": r["content"][:200],
                    "final_rank": r["final_rank"],
                    "similarity": r["similarity"],
                },
                memory_id=mid,
                conversation_id=conversation_id,
            )

    return results
