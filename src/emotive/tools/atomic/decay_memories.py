"""decay_memories: Manually run the decay process on all memories."""

from __future__ import annotations

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.queries.memory_queries import apply_decay


async def decay_memories_tool(
    ctx: Context,
    archive_threshold: float | None = None,
) -> dict:
    """Manually run the decay process on all memories.

    For testing forgetting dynamics. Applies the forgetting curve
    and archives memories below the retention threshold.

    Args:
        archive_threshold: Override the archive threshold (default from config).
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()
    threshold = archive_threshold or config.decay.archive_threshold

    session = app.session_factory()
    try:
        result = apply_decay(session, archive_threshold=threshold)
        session.commit()

        return {"status": "ok", "data": result}
    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "decay_failed", "message": str(e)}
    finally:
        session.close()
