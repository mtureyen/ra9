"""Shared memory operations: store and recall across all memory types."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import (
    apply_interference,
    create_memory_link,
    find_similar_memories,
    get_linked_memories,
    link_by_conversation,
    rank_memories,
    search_by_embedding,
    strengthen_link,
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
    """Store a memory with auto-generated embedding.

    Phase 0.5 brain-closer behaviors:
    - Instant linking: immediately find similar memories and link them
    - Temporal linking: link to other memories in same conversation
    - Interference: slightly weaken old memories that are very similar
    """
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

    # --- Brain-closer: instant linking on creation ---
    similar = find_similar_memories(
        session,
        embedding,
        threshold=0.65,
        exclude_ids=[mem.id],
        limit=5,
    )
    instant_links = 0
    for s in similar:
        result = create_memory_link(
            session, mem.id, s["id"],
            "semantic_similarity", strength=s["similarity"],
        )
        if result is not None:
            instant_links += 1

    # --- Brain-closer: temporal co-occurrence linking ---
    temporal_links = 0
    if conversation_id:
        temporal_links = link_by_conversation(
            session, conversation_id, strength=0.4,
        )

    # --- Brain-closer: interference-based forgetting ---
    interfered = apply_interference(
        session, embedding, mem.id,
        similarity_threshold=0.85,
        interference_strength=0.02,
    )

    if event_bus:
        event_bus.publish(
            MEMORY_STORED,
            {
                "memory_type": memory_type,
                "content": content[:200],
                "instant_links": instant_links,
                "temporal_links": temporal_links,
                "interfered_memories": interfered,
            },
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
    w_semantic: float = 0.4,
    w_recency: float = 0.25,
    w_activation: float = 0.2,
    w_significance: float = 0.15,
    spreading_hops: int = 2,
    spreading_decay: float = 0.6,
    event_bus: EventBus | None = None,
    conversation_id: uuid.UUID | None = None,
) -> list[dict]:
    """Full retrieval pipeline: embed -> search -> spread -> rank -> labile.

    Phase 0.5 brain-closer: co-recalled memories get linked together,
    strengthening associations between things accessed in the same context.
    """
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
        w_significance=w_significance,
    )

    # Step 4: Take top results, mark as labile, update retrieval stats
    results = ranked[:limit]
    now = datetime.now(timezone.utc)
    labile_until = now + timedelta(hours=1)

    result_ids = []
    for r in results:
        mid = r["id"]
        result_ids.append(mid)
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

    # --- Brain-closer: retrieval strengthens connections ---
    # Co-recalled memories get linked — co-activation = association
    if len(result_ids) >= 2:
        for i in range(len(result_ids)):
            for j in range(i + 1, min(i + 3, len(result_ids))):
                strengthen_link(
                    session,
                    result_ids[i],
                    result_ids[j],
                    "semantic_similarity",
                    boost=0.05,
                )

    return results
