"""consolidate: Run the memory consolidation engine."""

from __future__ import annotations

import uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.memory.consolidation import run_consolidation


async def consolidate_tool(
    ctx: Context,
    trigger_type: str = "manual",
    conversation_id: str | None = None,
) -> dict:
    """Run the consolidation engine — the 'sleep analog.'

    Promotes working memory to episodic, extracts semantic patterns,
    creates associative links, and applies memory decay.

    In Phase 0, no personality updates or identity checks are performed.

    Args:
        trigger_type: Why consolidation is running ("manual", "conversation_end", "hourly").
        conversation_id: Which conversation to consolidate (optional).
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()
    conv_id = uuid.UUID(conversation_id) if conversation_id else None

    session = app.session_factory()
    try:
        result = run_consolidation(
            session,
            app.embedding_service,
            config,
            conversation_id=conv_id,
            trigger_type=trigger_type,
            event_bus=app.event_bus,
        )
        session.commit()

        return {"status": "ok", "data": result}

    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "consolidation_failed", "message": str(e)}
    finally:
        session.close()
