"""Self-schema: the data structure representing Ryo's self-concept.

The SelfSchema is a weighted data structure — NOT prose. It's regenerated
(not retrieved) by the DMN analog, stored in RAM, and injected into every
LLM context.

Brain analog: mPFC abstract self-schemas as weighted patterns.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.memory.identity import load_identity_memories


@dataclass
class SelfSchema:
    """Weighted self-concept data structure."""

    traits: dict[str, float] = field(default_factory=dict)
    core_facts: list[str] = field(default_factory=list)
    active_values: list[str] = field(default_factory=list)
    person_context: dict[str, dict] = field(default_factory=dict)
    generated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def regenerate_schema(
    session: Session,
    *,
    max_traits: int = 10,
    max_core_facts: int = 10,
    max_values: int = 5,
) -> SelfSchema:
    """Regenerate self-schema from current memory state.

    Queries the same identity memories as begin_session, then derives
    traits from tag patterns and extracts person context.
    """
    # 1. Load identity memories (shared with begin_session)
    identity_mems = load_identity_memories(session, limit=20)

    # 2. Extract core facts from highest-significance memories
    core_facts = []
    for mem in identity_mems[:max_core_facts]:
        content = mem.get("content", "")
        if content and len(content) < 200:
            core_facts.append(content)

    # 3. Derive traits from tag frequency across all memories
    tag_counts = _get_tag_frequencies(session)
    # Normalize to 0-1 range
    max_count = max(tag_counts.values()) if tag_counts else 1
    traits = {
        tag: min(count / max_count, 1.0)
        for tag, count in tag_counts.most_common(max_traits)
        if tag not in ("gist", "conversation_summary", "conscious_intent")
    }

    # 4. Extract active values from high-significance semantic memories
    values = _extract_values(session, max_values)

    # 5. Extract person context from person-tagged memories
    person_context = _extract_person_context(session)

    return SelfSchema(
        traits=traits,
        core_facts=core_facts,
        active_values=values,
        person_context=person_context,
    )


def _get_tag_frequencies(session: Session) -> Counter:
    """Count tag frequencies across all non-archived memories."""
    stmt = (
        select(Memory.tags)
        .where(Memory.is_archived.is_(False))
        .where(func.array_length(Memory.tags, 1) > 0)
    )
    rows = session.execute(stmt).scalars().all()
    counter: Counter = Counter()
    for tags in rows:
        if tags:
            counter.update(tags)
    return counter


def _extract_values(session: Session, limit: int) -> list[str]:
    """Extract active values from high-significance semantic memories."""
    stmt = (
        select(Memory.content)
        .where(Memory.is_archived.is_(False))
        .where(Memory.memory_type == "semantic")
        .where(Memory.metadata_["significance"].as_float() >= 0.7)
        .order_by(Memory.metadata_["significance"].as_float().desc())
        .limit(limit)
    )
    rows = session.execute(stmt).scalars().all()
    values = []
    for content in rows:
        # Extract short value-like statements
        if content and len(content) < 150:
            values.append(content)
    return values


def _extract_person_context(session: Session) -> dict[str, dict]:
    """Extract person context from memories tagged with person names."""
    # Look for memories with person-related tags
    person_tags = {"personal", "relationship", "friend", "creator"}
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.tags.overlap(list(person_tags)))
        .order_by(Memory.retrieval_count.desc())
        .limit(20)
    )
    rows = session.execute(stmt).scalars().all()

    persons: dict[str, dict] = {}
    for mem in rows:
        content_lower = mem.content.lower()
        # Simple name extraction — look for common patterns
        for tag in (mem.tags or []):
            if tag not in person_tags and tag not in (
                "identity", "core", "memory", "gist", "conversation_summary"
            ):
                if tag not in persons:
                    persons[tag] = {"role": "known person", "mentions": 0}
                persons[tag]["mentions"] += 1

    return persons
