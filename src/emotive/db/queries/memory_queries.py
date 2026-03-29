"""Raw SQL queries for memory operations: vector search, spreading activation, decay."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import text
from sqlalchemy.orm import Session


def _embedding_to_sql(embedding: list[float]) -> str:
    """Convert embedding list to a Postgres vector literal string."""
    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def search_by_embedding(
    session: Session,
    embedding: list[float],
    *,
    limit: int = 20,
    memory_type: str | None = None,
    exclude_archived: bool = True,
) -> list[dict]:
    """Vector similarity search using pgvector cosine distance.

    Returns candidates with similarity scores (higher = more similar).
    Cosine distance is 1 - cosine_similarity, so we convert.
    """
    filters = []
    params: dict = {"embedding": _embedding_to_sql(embedding), "limit": limit}

    if exclude_archived:
        filters.append("is_archived = false")
    if memory_type is not None:
        filters.append("memory_type = :memory_type")
        params["memory_type"] = memory_type

    where = (" AND " + " AND ".join(filters)) if filters else ""

    sql = text(f"""
        SELECT
            id, memory_type, content, tags, metadata,
            confidence, reinforcement_count, detail_retention,
            is_formative, retrieval_count, created_at,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM memories
        WHERE true {where}
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    rows = session.execute(sql, params).mappings().all()
    return [dict(row) for row in rows]


def get_linked_memories(
    session: Session,
    memory_ids: list[uuid.UUID],
    *,
    hops: int = 2,
    decay_per_hop: float = 0.6,
) -> dict[uuid.UUID, float]:
    """Spreading activation: traverse memory links up to N hops.

    Returns a dict of {memory_id: activation_score} for memories
    reachable from the seed set. Seeds themselves are not included.
    """
    if not memory_ids:
        return {}

    seed_ids = [str(mid) for mid in memory_ids]
    activation: dict[uuid.UUID, float] = {}
    visited: set[str] = set(seed_ids)
    current_frontier = {str(mid): 1.0 for mid in memory_ids}

    for _hop in range(hops):
        if not current_frontier:
            break

        frontier_ids = list(current_frontier.keys())
        sql = text("""
            SELECT
                source_memory_id, target_memory_id, strength
            FROM memory_links
            WHERE source_memory_id = ANY(:ids)
               OR target_memory_id = ANY(:ids)
        """)
        rows = session.execute(sql, {"ids": frontier_ids}).mappings().all()

        next_frontier: dict[str, float] = {}
        for row in rows:
            source = str(row["source_memory_id"])
            target = str(row["target_memory_id"])
            strength = row["strength"]

            # Determine which end is the neighbor
            if source in current_frontier:
                neighbor = target
                parent_score = current_frontier[source]
            else:
                neighbor = source
                parent_score = current_frontier[target]

            if neighbor in visited:
                continue

            score = parent_score * strength * decay_per_hop
            # Keep the highest activation if reached from multiple paths
            if neighbor not in next_frontier or score > next_frontier[neighbor]:
                next_frontier[neighbor] = score

        for nid, score in next_frontier.items():
            mid = uuid.UUID(nid)
            if mid not in activation or score > activation[mid]:
                activation[mid] = score
            visited.add(nid)

        current_frontier = next_frontier

    return activation


def compute_recency_weight(created_at: datetime) -> float:
    """Recency weighting: 1 / (1 + days_elapsed^0.5)."""
    now = datetime.now(timezone.utc)
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)
    days = max((now - created_at).total_seconds() / 86400, 0)
    return 1.0 / (1.0 + days**0.5)


def rank_memories(
    candidates: list[dict],
    activation_scores: dict[uuid.UUID, float],
    *,
    w_semantic: float = 0.5,
    w_recency: float = 0.3,
    w_activation: float = 0.2,
) -> list[dict]:
    """Combine similarity, recency, and spreading activation into final ranking."""
    for mem in candidates:
        similarity = mem.get("similarity", 0.0)
        recency = compute_recency_weight(mem["created_at"])
        mid = mem["id"] if isinstance(mem["id"], uuid.UUID) else uuid.UUID(str(mem["id"]))
        spread = activation_scores.get(mid, 0.0)

        mem["recency_weight"] = recency
        mem["spreading_activation"] = spread
        mem["final_rank"] = (
            similarity * w_semantic
            + recency * w_recency
            + spread * w_activation
        )

    candidates.sort(key=lambda m: m["final_rank"], reverse=True)
    return candidates


def apply_decay(
    session: Session,
    *,
    archive_threshold: float = 0.1,
) -> dict[str, int]:
    """Apply forgetting curve to all non-archived memories.

    detail_retention = 1 / (1 + decay_rate * decay_protection * days_since_creation)

    Returns counts of decayed and archived memories.
    """
    # Update detail_retention based on elapsed time
    sql_decay = text("""
        UPDATE memories
        SET detail_retention = 1.0 / (
            1.0 + decay_rate * decay_protection
            * EXTRACT(EPOCH FROM (now() - created_at)) / 86400.0
        )
        WHERE is_archived = false
        RETURNING id, detail_retention
    """)
    result = session.execute(sql_decay).mappings().all()
    decayed_count = len(result)

    # Archive memories below threshold
    sql_archive = text("""
        UPDATE memories
        SET is_archived = true, archived_at = now()
        WHERE is_archived = false AND detail_retention < :threshold
        RETURNING id
    """)
    archived = session.execute(sql_archive, {"threshold": archive_threshold}).mappings().all()
    archived_count = len(archived)

    return {"memories_decayed": decayed_count, "memories_archived": archived_count}


def create_memory_link(
    session: Session,
    source_id: uuid.UUID,
    target_id: uuid.UUID,
    link_type: str,
    strength: float = 0.5,
    created_during: int | None = None,
) -> int | None:
    """Create a bidirectional memory link. Returns link id, or None if duplicate."""
    sql = text("""
        INSERT INTO memory_links
            (source_memory_id, target_memory_id, link_type, strength, created_during)
        VALUES (:source, :target, :link_type, :strength, :created_during)
        ON CONFLICT (source_memory_id, target_memory_id, link_type) DO NOTHING
        RETURNING id
    """)
    result = session.execute(sql, {
        "source": str(source_id),
        "target": str(target_id),
        "link_type": link_type,
        "strength": strength,
        "created_during": created_during,
    }).scalar_one_or_none()
    return result


def find_similar_memories(
    session: Session,
    embedding: list[float],
    *,
    threshold: float = 0.75,
    exclude_ids: list[uuid.UUID] | None = None,
    memory_type: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """Find memories above a similarity threshold. Used for consolidation clustering."""
    filters = ["is_archived = false"]
    emb_str = _embedding_to_sql(embedding)
    params: dict = {"embedding": emb_str, "threshold": threshold, "limit": limit}

    if exclude_ids:
        filters.append("id != ALL(:exclude_ids)")
        params["exclude_ids"] = [str(eid) for eid in exclude_ids]
    if memory_type:
        filters.append("memory_type = :memory_type")
        params["memory_type"] = memory_type

    where = " AND ".join(filters)

    sql = text(f"""
        SELECT
            id, memory_type, content, created_at,
            1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM memories
        WHERE {where}
          AND 1 - (embedding <=> CAST(:embedding AS vector)) >= :threshold
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    rows = session.execute(sql, params).mappings().all()
    return [dict(row) for row in rows]
