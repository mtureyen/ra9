"""store_memory: Directly store a memory into the database."""

from __future__ import annotations

import uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.memory.base import store_memory
from emotive.memory.episodic import EPISODIC_DECAY_RATE
from emotive.memory.procedural import PROCEDURAL_DECAY_RATE
from emotive.memory.semantic import SEMANTIC_DECAY_RATE

DECAY_RATES = {
    "episodic": EPISODIC_DECAY_RATE,
    "semantic": SEMANTIC_DECAY_RATE,
    "procedural": PROCEDURAL_DECAY_RATE,
}


async def store_memory_tool(
    ctx: Context,
    content: str,
    memory_type: str = "episodic",
    tags: list[str] | str | None = None,
    metadata: dict | None = None,
    significance: float | None = None,
    conversation_id: str | None = None,
) -> dict:
    """Directly store a memory. Generates embedding automatically.

    Args:
        content: The memory content to store.
        memory_type: "episodic", "semantic", or "procedural".
        tags: Optional semantic tags.
        metadata: Optional type-specific metadata (JSONB).
        significance: Optional importance score (0.0 to 1.0). Stored in metadata.
        conversation_id: Link to a conversation session.
    """
    app: AppContext = ctx.lifespan_context

    # Coerce tags from JSON string if LLM sends it that way
    if isinstance(tags, str):
        import json

        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = [tags]

    if memory_type not in ("episodic", "semantic", "procedural"):
        return {
            "status": "error",
            "error": "invalid_memory_type",
            "message": f"memory_type must be episodic/semantic/procedural, got {memory_type}",
        }

    if significance is not None and not (0.0 <= significance <= 1.0):
        return {
            "status": "error",
            "error": "invalid_significance",
            "message": f"significance must be between 0.0 and 1.0, got {significance}",
        }

    # Merge significance into metadata
    merged_metadata = dict(metadata or {})
    if significance is not None:
        merged_metadata["significance"] = significance

    conv_id = uuid.UUID(conversation_id) if conversation_id else None
    decay_rate = DECAY_RATES.get(memory_type, EPISODIC_DECAY_RATE)

    session = app.session_factory()
    try:
        mem = store_memory(
            session,
            app.embedding_service,
            content=content,
            memory_type=memory_type,
            conversation_id=conv_id,
            tags=tags,
            metadata=merged_metadata if merged_metadata else None,
            decay_rate=decay_rate,
            event_bus=app.event_bus,
        )
        session.commit()

        return {
            "status": "ok",
            "data": {
                "memory_id": str(mem.id),
                "memory_type": mem.memory_type,
                "content": mem.content[:200],
                "tags": mem.tags,
                "decay_rate": mem.decay_rate,
                "significance": significance,
            },
        }

    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "store_failed", "message": str(e)}
    finally:
        session.close()
