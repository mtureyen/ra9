"""export_timeline: Generate a timeline of the system's emotional evolution."""

from __future__ import annotations

from datetime import datetime

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.consolidation import ConsolidationLog
from emotive.db.models.episode import EmotionalEpisode
from emotive.db.models.memory import Memory


async def export_timeline_tool(
    ctx: Context,
    start: str | None = None,
    end: str | None = None,
    include: list[str] | None = None,
    limit: int = 100,
) -> dict:
    """Generate a timeline of the system's evolution.

    For visualization, analysis, and research.

    Args:
        start: ISO 8601 start time (optional).
        end: ISO 8601 end time (optional).
        include: What to include: "episodes", "memories", "consolidations".
                 Defaults to all.
        limit: Max entries per category (default 100).
    """
    app: AppContext = ctx.lifespan_context
    if include is None:
        include = ["episodes", "memories", "consolidations"]

    session = app.session_factory()
    try:
        timeline = {}

        if "episodes" in include:
            stmt = (
                select(EmotionalEpisode)
                .order_by(EmotionalEpisode.created_at.desc())
                .limit(limit)
            )
            if start:
                stmt = stmt.where(
                    EmotionalEpisode.created_at >= datetime.fromisoformat(start)
                )
            if end:
                stmt = stmt.where(
                    EmotionalEpisode.created_at <= datetime.fromisoformat(end)
                )
            episodes = session.execute(stmt).scalars().all()
            timeline["episodes"] = [
                {
                    "timestamp": str(ep.created_at),
                    "emotion": ep.primary_emotion,
                    "intensity": ep.intensity,
                    "is_formative": ep.is_formative,
                    "trigger": ep.trigger_event[:100],
                    "valence": ep.appraisal_valence,
                }
                for ep in episodes
            ]

        if "memories" in include:
            stmt = (
                select(Memory)
                .where(Memory.is_archived.is_(False))
                .order_by(Memory.created_at.desc())
                .limit(limit)
            )
            if start:
                stmt = stmt.where(
                    Memory.created_at >= datetime.fromisoformat(start)
                )
            if end:
                stmt = stmt.where(
                    Memory.created_at <= datetime.fromisoformat(end)
                )
            memories = session.execute(stmt).scalars().all()
            timeline["memories"] = [
                {
                    "timestamp": str(m.created_at),
                    "type": m.memory_type,
                    "content": m.content[:100],
                    "emotional_intensity": m.emotional_intensity,
                    "primary_emotion": m.primary_emotion,
                    "retention": m.detail_retention,
                }
                for m in memories
            ]

        if "consolidations" in include:
            stmt = (
                select(ConsolidationLog)
                .order_by(ConsolidationLog.started_at.desc())
                .limit(limit)
            )
            if start:
                stmt = stmt.where(
                    ConsolidationLog.started_at >= datetime.fromisoformat(start)
                )
            if end:
                stmt = stmt.where(
                    ConsolidationLog.started_at <= datetime.fromisoformat(end)
                )
            logs = session.execute(stmt).scalars().all()
            timeline["consolidations"] = [
                {
                    "timestamp": str(c.started_at),
                    "trigger": c.trigger_type,
                    "duration_ms": c.duration_ms,
                    "episodes_replayed": c.episodes_replayed,
                    "memories_promoted": c.memories_promoted,
                    "links_created": c.links_created,
                }
                for c in logs
            ]

        return {"status": "ok", "data": timeline}

    except Exception as e:
        return {
            "status": "error",
            "error": "export_failed",
            "message": str(e),
        }
    finally:
        session.close()
