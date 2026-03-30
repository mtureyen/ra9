"""Phase 0.5 consolidation engine: promote, extract, link, decay, concept hubs.

No personality updates or identity checks — those come in later phases.
"""

from __future__ import annotations

import time
import uuid
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.config.schema import EmotiveConfig
from emotive.db.models.consolidation import ConsolidationLog
from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import (
    apply_decay,
    create_memory_link,
    find_similar_memories,
)
from emotive.embeddings.service import EmbeddingService
from emotive.runtime.event_bus import (
    CONSOLIDATION_COMPLETED,
    CONSOLIDATION_STARTED,
    EventBus,
)
from emotive.runtime.working_memory import WorkingMemory

from .episodic import store_episodic
from .semantic import extract_semantic_from_cluster


def run_consolidation(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    working_memory: WorkingMemory | None = None,
    conversation_id: uuid.UUID | None = None,
    trigger_type: str = "conversation_end",
    event_bus: EventBus | None = None,
) -> dict:
    """Run the Phase 0.5 consolidation pipeline.

    Steps:
    1. Promote: working memory items above threshold -> episodic memory
    2. Extract: cluster similar episodics -> semantic memories
    3. Concept hubs: generate hub nodes from dense tag clusters
    4. Link: create associative links (new memories + all unlinked)
    5. Decay: apply forgetting curve, archive below threshold

    Returns a consolidation report dict.
    """
    start_time = time.time()

    log_entry = ConsolidationLog(
        trigger_type=trigger_type,
        conversation_id=conversation_id,
    )
    session.add(log_entry)
    session.flush()

    if event_bus:
        event_bus.publish(
            CONSOLIDATION_STARTED,
            {"trigger_type": trigger_type},
            consolidation_id=log_entry.id,
            conversation_id=conversation_id,
        )

    # Step 0: Episode replay (Phase 1+)
    episodes_replayed = 0
    if config.layers.episodes:
        episodes_replayed = _replay_episodes(
            session, embedding_service, config,
            conversation_id=conversation_id,
            event_bus=event_bus,
        )

    # Step 1: Promote
    promoted = _promote(
        session, embedding_service, config,
        working_memory=working_memory,
        conversation_id=conversation_id,
        event_bus=event_bus,
    )

    # Step 2: Extract
    extracted = _extract(
        session, embedding_service, config,
        event_bus=event_bus,
    )

    # Step 3: Concept hubs
    hubs_created = _build_concept_hubs(
        session, embedding_service, config,
        consolidation_id=log_entry.id,
        event_bus=event_bus,
    )

    # Step 4: Link (new memories + all unlinked)
    linked = _link_all(
        session, embedding_service, config,
        promoted_ids=[m.id for m in promoted],
        extracted_ids=[m.id for m in extracted],
        consolidation_id=log_entry.id,
    )

    # Step 5: Decay
    decay_result = apply_decay(
        session,
        archive_threshold=config.decay.archive_threshold,
    )

    # Update consolidation log
    duration_ms = int((time.time() - start_time) * 1000)
    log_entry.completed_at = datetime.now(timezone.utc)
    log_entry.duration_ms = duration_ms
    log_entry.episodes_replayed = episodes_replayed
    log_entry.memories_promoted = len(promoted)
    log_entry.patterns_extracted = len(extracted) + hubs_created
    log_entry.links_created = linked
    log_entry.memories_decayed = decay_result["memories_decayed"]
    log_entry.memories_archived = decay_result["memories_archived"]

    session.flush()

    if event_bus:
        event_bus.publish(
            CONSOLIDATION_COMPLETED,
            {
                "duration_ms": duration_ms,
                "memories_promoted": len(promoted),
                "patterns_extracted": len(extracted),
                "concept_hubs_created": hubs_created,
                "links_created": linked,
                "memories_decayed": decay_result["memories_decayed"],
                "episodes_replayed": episodes_replayed,
                "memories_archived": decay_result["memories_archived"],
            },
            consolidation_id=log_entry.id,
            conversation_id=conversation_id,
        )

    return {
        "consolidation_id": log_entry.id,
        "duration_ms": duration_ms,
        "replay": {
            "episodes_replayed": episodes_replayed,
        },
        "promotion": {
            "working_to_episodic": len(promoted),
        },
        "extraction": {
            "patterns_found": len(extracted),
            "concept_hubs_created": hubs_created,
            "new_semantic_memories": len(extracted) + hubs_created,
        },
        "linking": {
            "new_links_created": linked,
        },
        "decay": decay_result,
    }


def _promote(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    working_memory: WorkingMemory | None = None,
    conversation_id: uuid.UUID | None = None,
    event_bus: EventBus | None = None,
) -> list[Memory]:
    """Promote working memory items above significance threshold."""
    if working_memory is None:
        return []

    threshold = config.consolidation.significance_threshold
    significant = working_memory.get_above_threshold(threshold)
    promoted = []

    for item in significant:
        mem = store_episodic(
            session, embedding_service,
            content=item.content,
            conversation_id=conversation_id,
            tags=item.tags,
            context=item.metadata,
            event_bus=event_bus,
        )
        promoted.append(mem)

    return promoted


def _extract(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    event_bus: EventBus | None = None,
) -> list[Memory]:
    """Find clusters of similar episodic memories and extract semantic patterns."""
    stmt = (
        select(Memory)
        .where(Memory.memory_type == "episodic")
        .where(Memory.is_archived.is_(False))
        .where(Memory.consolidated_at.is_(None))
        .order_by(Memory.created_at.desc())
        .limit(100)
    )
    unconsolidated = list(session.execute(stmt).scalars().all())

    if len(unconsolidated) < config.consolidation.cluster_min_size:
        for m in unconsolidated:
            m.consolidated_at = m.created_at
        return []

    extracted = []
    clustered_ids: set[uuid.UUID] = set()

    for mem in unconsolidated:
        if mem.id in clustered_ids:
            continue

        similar = find_similar_memories(
            session,
            list(mem.embedding),
            threshold=config.consolidation.cluster_similarity_threshold,
            exclude_ids=list(clustered_ids),
            memory_type="episodic",
        )

        cluster_ids = [mem.id] + [
            s["id"] for s in similar if s["id"] != mem.id
        ]

        if len(cluster_ids) >= config.consolidation.cluster_min_size:
            cluster_memories = [
                session.get(Memory, cid)
                for cid in cluster_ids
                if session.get(Memory, cid) is not None
            ]
            semantic = extract_semantic_from_cluster(
                session, embedding_service, cluster_memories,
                event_bus=event_bus,
            )
            if semantic:
                extracted.append(semantic)
                clustered_ids.update(cluster_ids)

    now = datetime.now(timezone.utc)
    for m in unconsolidated:
        m.consolidated_at = now

    return extracted


def _build_concept_hubs(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    consolidation_id: int,
    event_bus: EventBus | None = None,
) -> int:
    """Generate concept hub nodes from dense tag clusters.

    When 5+ memories share a tag, create a semantic memory that acts as
    a hub node summarizing that concept. The hierarchy emerges from density.
    """
    # Get all active memories with tags
    from sqlalchemy import func

    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(func.array_length(Memory.tags, 1) > 0)
    )
    all_memories = list(session.execute(stmt).scalars().all())

    # Count tag frequency
    tag_counts: Counter[str] = Counter()
    tag_memories: dict[str, list[Memory]] = {}
    for m in all_memories:
        for tag in (m.tags or []):
            tag_counts[tag] += 1
            tag_memories.setdefault(tag, []).append(m)

    # Check which tags already have hub nodes
    existing_hubs = set()
    hub_stmt = (
        select(Memory)
        .where(Memory.memory_type == "semantic")
        .where(Memory.is_archived.is_(False))
    )
    for m in session.execute(hub_stmt).scalars().all():
        if m.metadata_ and m.metadata_.get("is_concept_hub"):
            existing_hubs.add(m.metadata_.get("hub_tag"))

    hubs_created = 0
    for tag, count in tag_counts.items():
        if count < 5 or tag in existing_hubs:
            continue

        # Build a summary from the tagged memories
        memories = tag_memories[tag]
        summary_parts = []
        for m in memories[:10]:
            summary_parts.append(m.content[:80])
        summary = f"Concept: {tag} — " + " | ".join(summary_parts)

        from .base import store_memory

        hub = store_memory(
            session, embedding_service,
            content=summary,
            memory_type="semantic",
            tags=[tag, "concept_hub"],
            metadata={
                "is_concept_hub": True,
                "hub_tag": tag,
                "member_count": count,
            },
            decay_rate=0.000001,
        )

        # Link hub to all members
        for m in memories:
            create_memory_link(
                session, hub.id, m.id,
                "conceptual_overlap", strength=0.6,
                created_during=consolidation_id,
            )

        hubs_created += 1

    return hubs_created


def _link_all(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    promoted_ids: list[uuid.UUID],
    extracted_ids: list[uuid.UUID],
    consolidation_id: int,
) -> int:
    """Create associative links — for new memories AND all unlinked memories."""
    from sqlalchemy import text

    # Find all memories with zero links
    sql = text("""
        SELECT m.id FROM memories m
        WHERE m.is_archived = false
          AND NOT EXISTS (
              SELECT 1 FROM memory_links ml
              WHERE ml.source_memory_id = m.id
                 OR ml.target_memory_id = m.id
          )
    """)
    unlinked_rows = session.execute(sql).mappings().all()
    unlinked_ids = [row["id"] for row in unlinked_rows]

    # Combine new + unlinked
    all_ids = list(set(promoted_ids + extracted_ids + unlinked_ids))
    if not all_ids:
        return 0

    links_created = 0
    threshold = config.consolidation.cluster_similarity_threshold

    for mid in all_ids:
        mem = session.get(Memory, mid)
        if mem is None:
            continue

        similar = find_similar_memories(
            session,
            list(mem.embedding),
            threshold=threshold,
            exclude_ids=[mid],
            limit=10,
        )

        for s in similar:
            result = create_memory_link(
                session, mid, s["id"],
                "semantic_similarity", strength=s["similarity"],
                created_during=consolidation_id,
            )
            if result is not None:
                links_created += 1

    return links_created


def _replay_episodes(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    conversation_id: uuid.UUID | None = None,
    event_bus: EventBus | None = None,
) -> int:
    """Replay unencoded episodes into episodic memory.

    Episodes that were created via experience_event but not yet encoded
    into memory get encoded here. This handles edge cases where the
    immediate encoding in experience_event failed.
    """
    from emotive.layers.episodes import get_unencoded_episodes

    unencoded = get_unencoded_episodes(session)
    if not unencoded:
        return 0

    from .episodic import store_episodic_from_episode

    replayed = 0
    for ep in unencoded:
        store_episodic_from_episode(
            session,
            embedding_service,
            episode=ep,
            content=ep.trigger_event,
            conversation_id=ep.conversation_id or conversation_id,
            tags=[ep.primary_emotion],
            encoding_strength_weight=config.episodes.encoding_strength_weight,
            event_bus=event_bus,
        )
        replayed += 1

    return replayed
