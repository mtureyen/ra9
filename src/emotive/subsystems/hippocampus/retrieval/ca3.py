"""CA3 pattern completion — the core retrieval engine.

Given a partial cue, converges to the nearest stored memory pattern
through attractor dynamics. Replaces one-shot cosine similarity with
iterative completion using the memory link network.

Algorithm:
  1. Start with query embedding (partial cue)
  2. Find K nearest memories by embedding similarity (seed set)
  3. For each seed, get linked memories (from memory_links)
  4. Compute combined activation: seed * 0.6 + mean(linked) * 0.4
  5. Re-query with completed pattern
  6. Repeat 2-3 times (attractor convergence)

The completed pattern is DIFFERENT from the original query — shaped
by the memory network's structure. This is how partial cues retrieve
memories the original embedding would miss.

Brain analog: CA3 autoassociative network, Hopfield attractor dynamics.
Sources: ScienceDirect S0896627313010854, Nature s41467-017-02752-1.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory, MemoryLink
from emotive.db.queries.memory_queries import search_by_embedding
from emotive.logging import get_logger

logger = get_logger("retrieval.ca3")


@dataclass
class CompletionCandidate:
    """A memory candidate from pattern completion."""

    memory_id: uuid.UUID
    content: str
    embedding: np.ndarray | None
    tags: list[str]
    completion_score: float  # how well it completed
    emotional_intensity: float
    primary_emotion: str | None
    memory_type: str
    created_at: object  # datetime
    retrieval_count: int
    is_formative: bool
    metadata: dict = field(default_factory=dict)
    # Raw similarity from initial search
    raw_similarity: float = 0.0


def pattern_complete(
    db_session: Session,
    query_embedding: list[float] | np.ndarray,
    *,
    iterations: int = 3,
    seed_limit: int = 15,
    emotional_boost: float = 1.3,
    mood_congruent_boost: float = 1.2,
    current_mood_valence: float | None = None,
    person_memory_ids: set[uuid.UUID] | None = None,
    exclude_ids: set[uuid.UUID] | None = None,
    rif_suppressed: dict[uuid.UUID, float] | None = None,
) -> list[CompletionCandidate]:
    """Run CA3 pattern completion on a query embedding.

    Args:
        db_session: DB session for queries.
        query_embedding: The retrieval cue (possibly DG-separated).
        iterations: Number of attractor convergence steps.
        seed_limit: How many initial candidates to fetch.
        emotional_boost: Multiplier for emotional memories (> 0.6 intensity).
        mood_congruent_boost: Multiplier for mood-matching memories.
        current_mood_valence: Current mood valence for congruence check.
        person_memory_ids: Pre-fetched IDs from person-node cache.
        exclude_ids: Memory IDs to exclude (already in consciousness).
        rif_suppressed: RIF suppression levels by memory ID.

    Returns:
        Ranked list of CompletionCandidates.
    """
    query = np.asarray(query_embedding, dtype=np.float32).tolist()
    exclude = exclude_ids or set()
    rif = rif_suppressed or {}

    # Phase 1: Direct retrieval — seed set from embedding similarity
    raw_results = search_by_embedding(
        db_session,
        query,
        limit=seed_limit,
    )

    # Merge person-node results (bypasses embedding)
    person_results = []
    if person_memory_ids:
        from sqlalchemy import select

        person_mems = (
            db_session.execute(
                select(Memory)
                .where(Memory.id.in_(person_memory_ids))
                .where(Memory.is_archived.is_(False))
                .limit(seed_limit)
            )
            .scalars()
            .all()
        )
        for m in person_mems:
            person_results.append(
                {
                    "id": m.id,
                    "content": m.content,
                    "embedding": list(m.embedding) if m.embedding is not None else None,
                    "tags": m.tags or [],
                    "similarity": 0.6,  # person-node match gets baseline score
                    "emotional_intensity": m.emotional_intensity or 0.0,
                    "primary_emotion": m.primary_emotion,
                    "memory_type": m.memory_type,
                    "created_at": m.created_at,
                    "retrieval_count": m.retrieval_count or 0,
                    "is_formative": m.is_formative or False,
                    "metadata": m.metadata_ or {},
                }
            )

    # Merge and dedupe
    seen_ids: set[uuid.UUID] = set()
    all_seeds: list[dict] = []
    for r in raw_results + person_results:
        mid = r["id"]
        if mid in seen_ids or mid in exclude:
            continue
        seen_ids.add(mid)
        all_seeds.append(r)

    if not all_seeds:
        return []

    # Phase 2: Iterative pattern completion
    current_seeds = all_seeds[:seed_limit]

    for iteration in range(iterations):
        # Get linked memories for current seeds
        seed_ids = [s["id"] for s in current_seeds[:10]]
        linked = _get_linked_embeddings(db_session, seed_ids)

        if not linked:
            break  # No links to follow, convergence

        # Compute combined pattern: seed * 0.6 + mean(linked) * 0.4
        seed_embeddings = [
            np.asarray(s["embedding"], dtype=np.float32)
            for s in current_seeds[:10]
            if s.get("embedding") is not None
        ]
        if not seed_embeddings:
            break

        seed_mean = np.mean(seed_embeddings, axis=0)
        linked_mean = np.mean(
            [np.asarray(e, dtype=np.float32) for e in linked],
            axis=0,
        )
        completed = 0.6 * seed_mean + 0.4 * linked_mean

        # Normalize
        norm = np.linalg.norm(completed)
        if norm > 1e-10:
            completed = completed / norm

        # Re-query with completed pattern
        new_results = search_by_embedding(
            db_session,
            completed.tolist(),
            limit=seed_limit,
        )

        # Merge new results into seed pool
        for r in new_results:
            mid = r["id"]
            if mid not in seen_ids and mid not in exclude:
                seen_ids.add(mid)
                all_seeds.append(r)

    # Phase 3: Score all candidates
    candidates = []
    for r in all_seeds:
        mid = r["id"]
        score = r.get("similarity", 0.5)

        # Emotional attractor boost
        intensity = r.get("emotional_intensity") or 0.0
        if intensity > 0.6:
            score *= emotional_boost

        # Mood-congruent boost
        if current_mood_valence is not None:
            emotion = r.get("primary_emotion", "")
            positive_emotions = {"joy", "trust", "awe", "surprise"}
            negative_emotions = {"sadness", "anger", "fear", "disgust"}
            if current_mood_valence > 0.55 and emotion in positive_emotions:
                score *= mood_congruent_boost
            elif current_mood_valence < 0.45 and emotion in negative_emotions:
                score *= mood_congruent_boost

        # E1: Deferred retrieval — near-miss memories from failed attempts get boost
        deferred = r.get("deferred_activation") or r.get("metadata", {}).get("deferred_activation", 0)
        if deferred and deferred > 0:
            score += float(deferred)

        # RIF suppression
        if mid in rif:
            score *= (1.0 - rif[mid])

        # Person-node bonus (tag-matched memories are more relevant)
        if person_memory_ids and mid in person_memory_ids:
            score *= 1.2

        candidates.append(
            CompletionCandidate(
                memory_id=mid,
                content=r.get("content", ""),
                embedding=(
                    np.asarray(r["embedding"], dtype=np.float32)
                    if r.get("embedding") is not None
                    else None
                ),
                tags=r.get("tags", []),
                completion_score=score,
                emotional_intensity=intensity,
                primary_emotion=r.get("primary_emotion"),
                memory_type=r.get("memory_type", "episodic"),
                created_at=r.get("created_at"),
                retrieval_count=r.get("retrieval_count", 0),
                is_formative=r.get("is_formative", False),
                metadata=r.get("metadata", {}),
                raw_similarity=r.get("similarity", 0.0),
            )
        )

    # Sort by completion score descending
    candidates.sort(key=lambda c: c.completion_score, reverse=True)
    return candidates


def _get_linked_embeddings(
    db_session: Session,
    seed_ids: list[uuid.UUID],
    max_per_seed: int = 5,
) -> list[list[float]]:
    """Get embeddings of memories linked to the seed set.

    Follows memory_links from seeds, returns linked memory embeddings
    for the pattern completion step.
    """
    from sqlalchemy import or_, select

    if not seed_ids:
        return []

    # Find linked memory IDs
    stmt = (
        select(MemoryLink)
        .where(
            or_(
                MemoryLink.source_memory_id.in_(seed_ids),
                MemoryLink.target_memory_id.in_(seed_ids),
            )
        )
        .where(MemoryLink.strength > 0.3)
        .order_by(MemoryLink.strength.desc())
        .limit(len(seed_ids) * max_per_seed)
    )
    links = db_session.execute(stmt).scalars().all()

    # Collect linked IDs (not in seed set)
    seed_set = set(seed_ids)
    linked_ids: set[uuid.UUID] = set()
    for link in links:
        if link.source_memory_id not in seed_set:
            linked_ids.add(link.source_memory_id)
        if link.target_memory_id not in seed_set:
            linked_ids.add(link.target_memory_id)

    if not linked_ids:
        return []

    # Fetch embeddings
    mems = (
        db_session.execute(
            select(Memory.embedding)
            .where(Memory.id.in_(linked_ids))
            .where(Memory.is_archived.is_(False))
            .where(Memory.embedding.isnot(None))
        )
        .scalars()
        .all()
    )

    return [list(e) for e in mems if e is not None]
