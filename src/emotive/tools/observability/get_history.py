"""get_history: Query historical data — consolidations, events."""

from __future__ import annotations

from datetime import datetime

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.consolidation import ConsolidationLog
from emotive.db.models.event_log import EventLog


async def get_history_tool(
    ctx: Context,
    history_type: str = "events",
    event_type: str | None = None,
    start: str | None = None,
    end: str | None = None,
    limit: int = 50,
) -> dict:
    """Query historical data: consolidation runs, event log, or episodes.

    Args:
        history_type: "events", "consolidations", or "episodes".
        event_type: Filter events by type (e.g., "memory_stored").
        start: ISO 8601 start time (optional).
        end: ISO 8601 end time (optional).
        limit: Max entries to return (default 50).
    """
    app: AppContext = ctx.lifespan_context

    session = app.session_factory()
    try:
        if history_type == "consolidations":
            return _query_consolidations(session, start, end, limit)
        elif history_type == "events":
            return _query_events(session, event_type, start, end, limit)
        elif history_type == "episodes":
            return _query_episodes(session, start, end, limit)
        else:
            return {
                "status": "error",
                "error": "invalid_history_type",
                "message": "history_type must be 'events', 'consolidations', or 'episodes'",
            }
    except Exception as e:
        return {"status": "error", "error": "get_history_failed", "message": str(e)}
    finally:
        session.close()


def _query_consolidations(
    session, start: str | None, end: str | None, limit: int
) -> dict:
    stmt = select(ConsolidationLog).order_by(ConsolidationLog.started_at.desc())
    if start:
        stmt = stmt.where(
            ConsolidationLog.started_at >= datetime.fromisoformat(start)
        )
    if end:
        stmt = stmt.where(
            ConsolidationLog.started_at <= datetime.fromisoformat(end)
        )
    stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()
    entries = []
    for r in rows:
        entries.append({
            "id": r.id,
            "trigger_type": r.trigger_type,
            "started_at": str(r.started_at),
            "duration_ms": r.duration_ms,
            "memories_promoted": r.memories_promoted,
            "patterns_extracted": r.patterns_extracted,
            "links_created": r.links_created,
            "memories_decayed": r.memories_decayed,
            "memories_archived": r.memories_archived,
        })

    return {"status": "ok", "data": {"history_type": "consolidations", "entries": entries}}


def _query_events(
    session, event_type: str | None, start: str | None, end: str | None, limit: int
) -> dict:
    stmt = select(EventLog).order_by(EventLog.recorded_at.desc())
    if event_type:
        stmt = stmt.where(EventLog.event_type == event_type)
    if start:
        stmt = stmt.where(EventLog.recorded_at >= datetime.fromisoformat(start))
    if end:
        stmt = stmt.where(EventLog.recorded_at <= datetime.fromisoformat(end))
    stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()
    entries = []
    for r in rows:
        entries.append({
            "id": r.id,
            "event_type": r.event_type,
            "event_data": r.event_data,
            "recorded_at": str(r.recorded_at),
            "memory_id": str(r.memory_id) if r.memory_id else None,
            "conversation_id": str(r.conversation_id) if r.conversation_id else None,
        })

    return {"status": "ok", "data": {"history_type": "events", "entries": entries}}


def _query_episodes(
    session, start: str | None, end: str | None, limit: int
) -> dict:
    from emotive.db.models.episode import EmotionalEpisode

    stmt = select(EmotionalEpisode).order_by(EmotionalEpisode.created_at.desc())
    if start:
        stmt = stmt.where(
            EmotionalEpisode.created_at >= datetime.fromisoformat(start)
        )
    if end:
        stmt = stmt.where(
            EmotionalEpisode.created_at <= datetime.fromisoformat(end)
        )
    stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()
    entries = []
    for r in rows:
        entries.append({
            "id": str(r.id),
            "trigger_event": r.trigger_event[:200],
            "primary_emotion": r.primary_emotion,
            "secondary_emotions": r.secondary_emotions,
            "intensity": r.intensity,
            "is_active": r.is_active,
            "is_formative": r.is_formative,
            "memory_encoded": r.memory_encoded,
            "appraisal": {
                "goal_relevance": r.appraisal_goal_relevance,
                "novelty": r.appraisal_novelty,
                "valence": r.appraisal_valence,
                "agency": r.appraisal_agency,
                "social_significance": r.appraisal_social_significance,
            },
            "created_at": str(r.created_at),
        })

    return {"status": "ok", "data": {"history_type": "episodes", "entries": entries}}
