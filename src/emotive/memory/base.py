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

# E6: Cache for formation period count (avoids SELECT COUNT on every store)
_formation_period_cache: int | None = None


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
    emotional_intensity: float | None = None,
    primary_emotion: str | None = None,
    valence: float | None = None,
    source_episode_id: uuid.UUID | None = None,
    decay_protection: float | None = None,
    event_bus: EventBus | None = None,
    # Phase Anamnesis fields
    encoding_mood: dict | None = None,
    source_type: str | None = None,
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
    if emotional_intensity is not None:
        mem.emotional_intensity = emotional_intensity
    if primary_emotion is not None:
        mem.primary_emotion = primary_emotion
    if valence is not None:
        mem.valence = valence
    if source_episode_id is not None:
        mem.source_episode_id = source_episode_id
    if decay_protection is not None:
        mem.decay_protection = decay_protection

    # Phase Anamnesis: encoding-time fields
    # E10: Mood-dependent gating — snapshot mood at encoding
    if encoding_mood:
        mem.encoding_mood = encoding_mood
    # E9: Store original embedding for drift tracking
    mem.original_embedding = embedding
    # E26: Imagination inflation guard — permanent source type
    if source_type:
        if mem.metadata_ is None:
            mem.metadata_ = {}
        mem.metadata_["source_type"] = source_type
    # E6: Formation period — first 150 memories get flagged
    # Use cached count to avoid SELECT COUNT(*) on every store
    global _formation_period_cache
    if _formation_period_cache is None or _formation_period_cache < 150:
        from sqlalchemy import func, select
        total = session.execute(
            select(func.count()).select_from(Memory).where(Memory.is_archived.is_(False))
        ).scalar() or 0
        _formation_period_cache = total
    if _formation_period_cache < 150:
        mem.formation_period = True
        _formation_period_cache += 1  # approximate, avoids recount

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

    # --- E12: Retroactive interference protection ---
    # When a new memory is very similar to existing ones from different
    # people/contexts, boost the existing memory's distinctiveness.
    try:
        from emotive.subsystems.hippocampus.retrieval.interference import (
            protect_against_retroactive,
        )
        protect_against_retroactive(session, embedding, mem.id)
    except Exception:
        pass  # Non-critical — don't crash encoding

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
    query_embedding: list[float] | None = None,
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

    If query_embedding is provided, skips the embed step (used by the
    thalamus to share a single embedding across subsystems).
    """
    if query_embedding is None:
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

    # Publish recall events AFTER the loop to avoid opening DB handler
    # sessions for each memory while still inside the caller's session scope.
    # Batch into a single event to reduce connection pressure.
    if event_bus and results:
        event_bus.publish(
            MEMORY_RECALLED,
            {
                "count": len(results),
                "top_content": results[0]["content"][:200] if results else "",
                "top_similarity": results[0].get("similarity", 0) if results else 0,
            },
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

    # --- Reconsolidation: update tags on already-labile memories ---
    # When a memory was already labile (recalled recently) and is recalled
    # again in a new context, it picks up tags from the co-active memories.
    # This mirrors how human memories gain new meaning when recalled in
    # different contexts.
    _EXCLUDED_TAGS = {"gist", "conversation_summary", "conscious_intent"}
    all_context_tags: set[str] = set()
    for r in results:
        all_context_tags.update(t for t in (r.get("tags") or []) if t not in _EXCLUDED_TAGS)

    for r in results:
        was_already_labile = r.get("is_labile", False)
        if was_already_labile and all_context_tags:
            existing = set(r.get("tags") or [])
            new_tags = [t for t in all_context_tags if t not in existing]
            if new_tags:
                updated = list(existing) + new_tags[:3]
                session.execute(
                    Memory.__table__.update()
                    .where(Memory.__table__.c.id == r["id"])
                    .values(tags=updated)
                )

    return results
