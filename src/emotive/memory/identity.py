"""Identity memory queries — shared between begin_session and DMN subsystem."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory


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
