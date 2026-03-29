"""Phase 0 consolidation engine: promote, extract, link, decay.

No personality updates or identity checks — those come in later phases.
"""

from __future__ import annotations

import time
import uuid

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
    """Run the Phase 0 consolidation pipeline.

    Steps:
    1. Promote: working memory items above threshold → episodic memory
    2. Extract: cluster similar episodics → semantic memories
    3. Link: create associative links between related memories
    4. Decay: apply forgetting curve, archive below threshold

    Returns a consolidation report dict.
    """
    start_time = time.time()

    # Create consolidation log entry
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

    # Step 1: Promote
    promoted = _promote(
        session,
        embedding_service,
        config,
        working_memory=working_memory,
        conversation_id=conversation_id,
        event_bus=event_bus,
    )

    # Step 2: Extract
    extracted = _extract(
        session,
        embedding_service,
        config,
        event_bus=event_bus,
    )

    # Step 3: Link
    linked = _link(
        session,
        embedding_service,
        config,
        promoted_ids=[m.id for m in promoted],
        extracted_ids=[m.id for m in extracted],
        consolidation_id=log_entry.id,
    )

    # Step 4: Decay
    decay_result = apply_decay(
        session,
        archive_threshold=config.decay.archive_threshold,
    )

    # Update consolidation log
    duration_ms = int((time.time() - start_time) * 1000)
    from datetime import datetime, timezone

    log_entry.completed_at = datetime.now(timezone.utc)
    log_entry.duration_ms = duration_ms
    log_entry.memories_promoted = len(promoted)
    log_entry.patterns_extracted = len(extracted)
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
                "links_created": linked,
                "memories_decayed": decay_result["memories_decayed"],
                "memories_archived": decay_result["memories_archived"],
            },
            consolidation_id=log_entry.id,
            conversation_id=conversation_id,
        )

    return {
        "consolidation_id": log_entry.id,
        "duration_ms": duration_ms,
        "promotion": {
            "working_to_episodic": len(promoted),
        },
        "extraction": {
            "patterns_found": len(extracted),
            "new_semantic_memories": len(extracted),
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
    """Promote working memory items above significance threshold to episodic memory."""
    if working_memory is None:
        return []

    threshold = config.consolidation.significance_threshold
    significant = working_memory.get_above_threshold(threshold)
    promoted = []

    for item in significant:
        mem = store_episodic(
            session,
            embedding_service,
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
    # Get all non-archived episodic memories that haven't been consolidated yet
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
        # Mark them as consolidated even if no pattern extracted
        for m in unconsolidated:
            m.consolidated_at = m.created_at  # placeholder
        return []

    extracted = []
    clustered_ids: set[uuid.UUID] = set()

    for mem in unconsolidated:
        if mem.id in clustered_ids:
            continue

        # Find similar memories to this one
        similar = find_similar_memories(
            session,
            list(mem.embedding),
            threshold=config.consolidation.cluster_similarity_threshold,
            exclude_ids=list(clustered_ids),
            memory_type="episodic",
        )

        # Build cluster (include the seed memory)
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
                session,
                embedding_service,
                cluster_memories,
                event_bus=event_bus,
            )
            if semantic:
                extracted.append(semantic)
                clustered_ids.update(cluster_ids)

    # Mark all unconsolidated as consolidated
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    for m in unconsolidated:
        m.consolidated_at = now

    return extracted


def _link(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    promoted_ids: list[uuid.UUID],
    extracted_ids: list[uuid.UUID],
    consolidation_id: int,
) -> int:
    """Create associative links for newly created memories."""
    new_ids = promoted_ids + extracted_ids
    if not new_ids:
        return 0

    links_created = 0
    threshold = config.consolidation.cluster_similarity_threshold

    for mid in new_ids:
        mem = session.get(Memory, mid)
        if mem is None:
            continue

        # Find similar existing memories
        similar = find_similar_memories(
            session,
            list(mem.embedding),
            threshold=threshold,
            exclude_ids=[mid],
            limit=10,
        )

        for s in similar:
            result = create_memory_link(
                session,
                mid,
                s["id"],
                "semantic_similarity",
                strength=s["similarity"],
                created_during=consolidation_id,
            )
            if result is not None:
                links_created += 1

    return links_created
