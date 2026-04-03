"""Shared logic for closing orphaned sessions."""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.config.schema import EmotiveConfig
from emotive.db.models.conversation import Conversation
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.runtime.event_bus import EventBus

from .consolidation import run_consolidation

logger = get_logger("session_cleanup")


def close_orphaned_sessions(
    session: Session,
    embedding_service: EmbeddingService,
    config: EmotiveConfig,
    *,
    event_bus: EventBus | None = None,
    limit: int = 10,
) -> int:
    """Find and close any sessions that were never properly ended.

    Runs consolidation on each orphan, then sets ended_at.
    Returns the number of sessions cleaned up.
    """
    stmt = (
        select(Conversation)
        .where(Conversation.ended_at.is_(None))
        .order_by(Conversation.started_at.asc())
        .limit(limit)
    )
    orphans = list(session.execute(stmt).scalars().all())

    if not orphans:
        return 0

    cleaned = 0
    for conv in orphans:
        try:
            conv.ended_at = datetime.now(timezone.utc)
            session.flush()

            if config.consolidation.auto_on_session_end:
                run_consolidation(
                    session,
                    embedding_service,
                    config,
                    conversation_id=conv.id,
                    trigger_type="orphan_cleanup",
                    event_bus=event_bus,
                )

            cleaned += 1
        except Exception as e:
            logger.warning("Failed to clean orphan session %s: %s", conv.id, e)
            continue

    return cleaned
