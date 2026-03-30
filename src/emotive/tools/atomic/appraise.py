"""appraise: Run the appraisal engine without creating an episode."""

from __future__ import annotations

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.models.temperament import Temperament
from emotive.layers.appraisal import rule_based_appraisal, run_appraisal


async def appraise_tool(
    ctx: Context,
    event: str,
    source: str = "user_message",
) -> dict:
    """Run the appraisal engine on an event without side effects.

    Returns the appraisal vector, emotion, and intensity — but does NOT
    create an episode or encode a memory. For testing and calibration.

    Args:
        event: Description of what happened.
        source: "user_message", "internal_realization", or "memory_retrieval".
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()

    if not config.layers.episodes:
        return {
            "status": "error",
            "error": "tool_not_available",
            "message": (
                "appraise requires Phase 1+ (emotional episodes). "
                f"Current phase: {config.phase}"
            ),
        }

    session = app.session_factory()
    try:
        temp = session.get(Temperament, 1)
        sensitivity = temp.sensitivity if temp else 0.5
        resilience = temp.resilience if temp else 0.5

        vector = rule_based_appraisal(event, source)
        result = run_appraisal(
            vector,
            sensitivity=sensitivity,
            resilience=resilience,
            base_half_life=config.episodes.base_half_life_minutes,
            formative_threshold=config.episodes.formative_intensity_threshold,
        )

        return {
            "status": "ok",
            "data": {
                "appraisal": {
                    "goal_relevance": vector.goal_relevance,
                    "novelty": vector.novelty,
                    "valence": vector.valence,
                    "agency": vector.agency,
                    "social_significance": vector.social_significance,
                },
                "primary_emotion": result.primary_emotion,
                "secondary_emotions": result.secondary_emotions,
                "intensity": round(result.intensity, 4),
                "half_life_minutes": round(result.half_life_minutes, 1),
                "is_formative": result.is_formative,
                "note": "Dry run — no episode created, no memory stored",
            },
        }
    finally:
        session.close()
