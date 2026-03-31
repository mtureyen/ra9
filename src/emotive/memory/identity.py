"""Identity memory queries — shared between begin_session and DMN subsystem."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.logging import get_logger

logger = get_logger("memory.identity")


def load_identity_memories(session: Session, limit: int = 10) -> list[dict]:
    """Load the most important memories for identity continuity.

    Pulls memories by: highest retrieval count, highest significance,
    and formative status. These are the memories that define who you are.
    """
    # Most retrieved (identity anchors — what gets recalled every session)
    most_retrieved = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.retrieval_count > 0)
        .order_by(Memory.retrieval_count.desc())
        .limit(5)
    )

    # Highest significance
    high_sig = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.metadata_["significance"].as_float() >= 0.8)
        .order_by(Memory.metadata_["significance"].as_float().desc())
        .limit(5)
    )

    # Formative memories
    formative = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.is_formative.is_(True))
        .limit(5)
    )

    # Combine and deduplicate
    seen_ids = set()
    results = []

    for stmt in [most_retrieved, high_sig, formative]:
        rows = session.execute(stmt).scalars().all()
        for m in rows:
            if m.id not in seen_ids and len(results) < limit:
                seen_ids.add(m.id)
                sig = None
                if m.metadata_ and "significance" in m.metadata_:
                    sig = m.metadata_["significance"]
                results.append({
                    "id": str(m.id),
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "significance": sig,
                    "retrieval_count": m.retrieval_count,
                    "is_formative": m.is_formative,
                    "tags": m.tags or [],
                })

    return results


def compute_person_trust(session: Session, person_name: str) -> str:
    """Derive trust level from accumulated interaction data.

    Returns: 'unknown', 'familiar', 'trusted', or 'core'.

    Logic:
    - Count memories where content contains the person's name
    - Count formative memories mentioning them
    - Get avg retrieval count of memories mentioning them
    - "unknown": 0 mentions
    - "familiar": 1-4 mentions
    - "trusted": 5+ mentions with positive content ratio
    - "core": formative memories + 10+ mentions OR retrieval avg > 3
    """
    name_lower = person_name.lower()

    # Find all non-archived memories mentioning this person
    all_mems = (
        session.execute(
            select(Memory).where(Memory.is_archived.is_(False))
        )
        .scalars()
        .all()
    )

    # Filter by name (case-insensitive content match)
    matching = [m for m in all_mems if name_lower in (m.content or "").lower()]
    mention_count = len(matching)

    if mention_count == 0:
        return "unknown"

    # Count formative mentions
    formative_count = sum(1 for m in matching if m.is_formative)

    # Average retrieval count
    avg_retrieval = (
        sum(m.retrieval_count or 0 for m in matching) / mention_count
    )

    # Core: formative memories + 10+ mentions OR retrieval avg > 3
    if (formative_count > 0 and mention_count >= 10) or avg_retrieval > 3:
        return "core"

    # Trusted: 5+ mentions
    if mention_count >= 5:
        return "trusted"

    # Familiar: 1-4 mentions
    return "familiar"
