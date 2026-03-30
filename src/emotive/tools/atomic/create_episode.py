"""create_episode: Manually create an episode with explicit appraisal values."""

from __future__ import annotations

import uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.models.temperament import Temperament
from emotive.layers.appraisal import AppraisalVector, run_appraisal
from emotive.layers.episodes import create_episode


async def create_episode_tool(
    ctx: Context,
    trigger_event: str,
    goal_relevance: float,
    novelty: float,
    valence: float,
    agency: float,
    social_significance: float,
    source: str = "manual",
    conversation_id: str | None = None,
) -> dict:
    """Manually create an episode with specified appraisal values.

    Bypasses the appraisal engine. For testing specific emotional configs.

    Args:
        trigger_event: What happened.
        goal_relevance: 0-1.
        novelty: 0-1.
        valence: 0-1 (0=negative, 1=positive).
        agency: 0-1 (0=external, 1=self-caused).
        social_significance: 0-1.
        source: Event source type.
        conversation_id: Optional session link.
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()

    if not config.layers.episodes:
        return {
            "status": "error",
            "error": "tool_not_available",
            "message": (
                "create_episode requires Phase 1+ (emotional episodes). "
                f"Current phase: {config.phase}"
            ),
        }

    session = app.session_factory()
    try:
        temp = session.get(Temperament, 1)
        sensitivity = temp.sensitivity if temp else 0.5
        resilience = temp.resilience if temp else 0.5

        vector = AppraisalVector(
            goal_relevance=goal_relevance,
            novelty=novelty,
            valence=valence,
            agency=agency,
            social_significance=social_significance,
        )

        result = run_appraisal(
            vector,
            sensitivity=sensitivity,
            resilience=resilience,
            base_half_life=config.episodes.base_half_life_minutes,
            formative_threshold=config.episodes.formative_intensity_threshold,
        )

        conv_id = uuid.UUID(conversation_id) if conversation_id else None

        episode = create_episode(
            session,
            result,
            trigger_event=trigger_event,
            trigger_source=source,
            conversation_id=conv_id,
            event_bus=app.event_bus,
        )

        session.commit()

        return {
            "status": "ok",
            "data": {
                "episode_id": str(episode.id),
                "primary_emotion": result.primary_emotion,
                "secondary_emotions": result.secondary_emotions,
                "intensity": round(result.intensity, 4),
                "is_formative": result.is_formative,
            },
        }

    except Exception as e:
        session.rollback()
        return {
            "status": "error",
            "error": "create_episode_failed",
            "message": str(e),
        }
    finally:
        session.close()
