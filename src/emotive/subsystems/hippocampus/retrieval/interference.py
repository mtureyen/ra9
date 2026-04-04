"""Proactive and retroactive interference — E12.

Two asymmetric memory competition patterns:

Retroactive (new disrupts old): operates at ENCODING time.
  New learning partially reactivates old traces, degrading
  the original's distinctiveness. Appears immediately.

Proactive (old disrupts new): operates at RETRIEVAL time.
  Residual activation of old associations competes with newer
  ones during retrieval. Emerges after a delay.

Brain analog: DG + CA3 interference patterns.
Sources: Alves et al. (2025), MDPI; Bhatt et al. (2019), Nature.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import find_similar_memories
from emotive.logging import get_logger

logger = get_logger("hippocampus.retrieval.interference")


def protect_against_retroactive(
    db_session: Session,
    new_embedding: list[float],
    new_memory_id: uuid.UUID,
    threshold: float = 0.8,
) -> int:
    """Retroactive interference protection at encoding time.

    When encoding a new memory that's highly similar to existing ones
    (from different people/contexts), boost the existing memory's
    distinctiveness to prevent the new memory from burying it.

    Called during store_memory after instant linking.
    Returns number of memories protected.
    """
    similar = find_similar_memories(
        db_session,
        new_embedding,
        threshold=threshold,
        exclude_ids=[new_memory_id],
        limit=5,
    )

    protected = 0
    for s in similar:
        mem = db_session.get(Memory, s["id"])
        if mem is None:
            continue

        # Only protect if different person/context (same person = natural update)
        new_mem = db_session.get(Memory, new_memory_id)
        if new_mem and _same_person(new_mem, mem):
            continue

        # Boost existing memory's activation to resist being buried
        mem.current_activation = min(
            1.0, (mem.current_activation or 0.5) + 0.1,
        )
        protected += 1

    if protected:
        db_session.flush()
        logger.info("Retroactive protection: %d memories boosted", protected)

    return protected


def detect_proactive_interference(
    candidates: list,
    now: datetime | None = None,
) -> list:
    """Proactive interference detection at retrieval time.

    When a high-activation OLD memory competes with a lower-activation
    NEWER memory on the same topic, reduce the old memory's competition
    weight. This prevents outdated understanding from permanently
    blocking updated understanding.

    Operates on CompletionCandidate or ComparatorResult objects.
    Returns the same list with scores adjusted.
    """
    if now is None:
        now = datetime.now(timezone.utc)

    if len(candidates) < 2:
        return candidates

    for i, newer in enumerate(candidates):
        c_new = newer.candidate if hasattr(newer, 'candidate') else newer
        new_created = getattr(c_new, 'created_at', None)
        if not new_created:
            continue

        for j, older in enumerate(candidates):
            if i == j:
                continue
            c_old = older.candidate if hasattr(older, 'candidate') else older
            old_created = getattr(c_old, 'created_at', None)
            if not old_created:
                continue

            # Check: is old memory significantly older AND higher activation?
            if hasattr(old_created, 'timestamp'):
                old_created = old_created
            if hasattr(new_created, 'timestamp'):
                new_created = new_created

            try:
                age_diff_days = (new_created - old_created).days
            except (TypeError, AttributeError):
                continue

            if age_diff_days < 1:
                continue  # not old enough for proactive interference

            old_score = getattr(older, 'final_score', getattr(older, 'completion_score', 0))
            new_score = getattr(newer, 'final_score', getattr(newer, 'completion_score', 0))

            if old_score > new_score * 1.5:
                # Proactive interference: old memory blocking newer
                # Reduce old memory's competition weight
                if hasattr(older, 'final_score'):
                    older.final_score *= 0.7
                elif hasattr(older, 'completion_score'):
                    older.completion_score *= 0.7

                logger.debug(
                    "Proactive interference: old memory (%.0f days) reduced to let newer through",
                    age_diff_days,
                )

    return candidates


def _same_person(mem_a: Memory, mem_b: Memory) -> bool:
    """Check if two memories are about the same person."""
    tags_a = set(mem_a.tags or [])
    tags_b = set(mem_b.tags or [])

    # Known non-person tags
    _SYSTEM = {
        "joy", "trust", "awe", "surprise", "sadness", "anger",
        "fear", "disgust", "neutral", "curiosity", "contempt",
        "episodic", "semantic", "procedural", "gist",
        "inner_speech", "contradiction", "behavioral_coaching",
    }

    persons_a = tags_a - _SYSTEM
    persons_b = tags_b - _SYSTEM

    return bool(persons_a & persons_b)
