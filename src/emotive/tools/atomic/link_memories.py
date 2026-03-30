"""link_memories: Manually create an associative link between two memories."""

from __future__ import annotations

import uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.queries.memory_queries import create_memory_link


async def link_memories_tool(
    ctx: Context,
    source_memory_id: str,
    target_memory_id: str,
    link_type: str = "semantic_similarity",
    strength: float = 0.5,
) -> dict:
    """Manually create an associative link between two memories.

    Args:
        source_memory_id: UUID of the source memory.
        target_memory_id: UUID of the target memory.
        link_type: One of: semantic_similarity, temporal_proximity,
                   conceptual_overlap, emotional_resonance, causal, contradiction.
        strength: Link strength (0.0 to 1.0).
    """
    app: AppContext = ctx.lifespan_context

    valid_types = {
        "semantic_similarity", "temporal_proximity", "conceptual_overlap",
        "emotional_resonance", "causal", "contradiction",
    }
    if link_type not in valid_types:
        return {
            "status": "error",
            "error": "invalid_link_type",
            "message": f"link_type must be one of {valid_types}",
        }

    session = app.session_factory()
    try:
        link_id = create_memory_link(
            session,
            uuid.UUID(source_memory_id),
            uuid.UUID(target_memory_id),
            link_type,
            strength=strength,
        )
        session.commit()

        if link_id is None:
            return {
                "status": "ok",
                "data": {"created": False, "note": "Link already exists"},
            }

        return {
            "status": "ok",
            "data": {
                "created": True,
                "link_id": link_id,
                "link_type": link_type,
                "strength": strength,
            },
        }
    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "link_failed", "message": str(e)}
    finally:
        session.close()
