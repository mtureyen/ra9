"""end_session: Close a conversation and optionally trigger consolidation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.conversation import Conversation
from emotive.db.models.temperament import Temperament
from emotive.memory.consolidation import run_consolidation
from emotive.runtime.event_bus import SESSION_ENDED


async def end_session_tool(
    ctx: Context,
    conversation_id: str | None = None,
    summary: str | None = None,
    topics: list[str] | None = None,
    trigger_consolidation: bool = True,
) -> dict:
    """End a conversation session. Optionally triggers memory consolidation.

    Call this once at the end of every conversation.
    If conversation_id is not provided, closes the most recent open session.
    """
    app: AppContext = ctx.lifespan_context

    session = app.session_factory()
    try:
        # Resolve conversation ID
        if conversation_id:
            conv_id = uuid.UUID(conversation_id)
        else:
            stmt = (
                select(Conversation)
                .where(Conversation.ended_at.is_(None))
                .order_by(Conversation.started_at.desc())
                .limit(1)
            )
            latest = session.execute(stmt).scalar_one_or_none()
            if latest is None:
                return {
                    "status": "error",
                    "error": "no_open_session",
                    "message": "No open conversation to end",
                }
            conv_id = latest.id

        conv = session.get(Conversation, conv_id)
        if conv is None:
            return {
                "status": "error",
                "error": "conversation_not_found",
                "message": f"No conversation with id {conv_id}",
            }

        # Close conversation
        conv.ended_at = datetime.now(timezone.utc)
        if summary:
            conv.summary = summary
        if topics:
            conv.topics = topics
        session.flush()

        # Calculate duration
        duration_minutes = None
        if conv.started_at:
            delta = conv.ended_at - conv.started_at
            duration_minutes = round(delta.total_seconds() / 60, 1)

        # Archive decayed episodes (Phase 1+)
        episodes_archived = 0
        config = app.config_manager.get()
        if config.layers.episodes:
            from emotive.layers.episodes import archive_decayed_episodes

            temp = session.get(Temperament, 1)
            sensitivity = temp.sensitivity if temp else 0.5
            episodes_archived = archive_decayed_episodes(
                session, sensitivity, event_bus=app.event_bus,
            )

        # Consolidation
        consolidation_result = None
        if trigger_consolidation:
            if config.consolidation.auto_on_session_end:
                consolidation_result = run_consolidation(
                    session,
                    app.embedding_service,
                    config,
                    conversation_id=conv_id,
                    trigger_type="conversation_end",
                    event_bus=app.event_bus,
                )

        app.event_bus.publish(
            SESSION_ENDED,
            {
                "duration_minutes": duration_minutes,
                "message_count": conv.message_count,
                "consolidation_triggered": consolidation_result is not None,
            },
            conversation_id=conv_id,
        )

        session.commit()

        result = {
            "conversation_id": str(conv_id),
            "duration_minutes": duration_minutes,
            "message_count": conv.message_count,
            "episodes_archived": episodes_archived,
        }
        if consolidation_result:
            result["consolidation"] = consolidation_result

        return {"status": "ok", "data": result}

    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "session_end_failed", "message": str(e)}
    finally:
        session.close()
