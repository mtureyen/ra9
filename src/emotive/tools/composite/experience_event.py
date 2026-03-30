"""experience_event: Process an emotional event through the appraisal engine."""

from __future__ import annotations

import uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.models.temperament import Temperament
from emotive.layers.appraisal import (
    AppraisalVector,
    rule_based_appraisal,
    run_appraisal,
)
from emotive.layers.episodes import create_episode
from emotive.memory.episodic import store_episodic_from_episode


async def experience_event_tool(
    ctx: Context,
    event: str | None = None,
    source: str = "user_message",
    description: str | None = None,
    conversation_id: str | None = None,
    appraisal: dict | str | None = None,
    context: str | None = None,
) -> dict:
    """Process an emotional event. Generates an episode and encodes to memory.

    The core Phase 1 tool. Evaluates what happened emotionally, creates
    an episode, and stores it as an emotionally-tagged memory.

    Args:
        event: Description of what happened.
        description: Alias for event.
        source: "user_message", "internal_realization", or "memory_retrieval".
        conversation_id: Link to current session.
        appraisal: Optional self-assessed appraisal vector with keys:
            goal_relevance, novelty, valence, agency, social_significance (all 0-1).
            If not provided, a rule-based fallback is used.
        context: Optional additional context about the event.
    """
    app: AppContext = ctx.lifespan_context

    # Accept "description" as alias for "event"
    if event is None and description is not None:
        event = description
    if event is None:
        return {
            "status": "error",
            "error": "missing_event",
            "message": "event (or description) is required",
        }

    # Coerce appraisal from JSON string if needed
    if isinstance(appraisal, str):
        import json

        try:
            appraisal = json.loads(appraisal)
        except (json.JSONDecodeError, TypeError):
            appraisal = None

    config = app.config_manager.get()

    if not config.layers.episodes:
        return {
            "status": "error",
            "error": "tool_not_available",
            "message": (
                "experience_event requires Phase 1+ (emotional episodes). "
                f"Current phase: {config.phase}"
            ),
            "current_phase": config.phase,
            "required_phase": 1,
        }

    session = app.session_factory()
    try:
        # Read temperament
        temp = session.get(Temperament, 1)
        sensitivity = temp.sensitivity if temp else 0.5
        resilience = temp.resilience if temp else 0.5

        # Build appraisal vector
        if appraisal:
            try:
                vector = AppraisalVector(
                    goal_relevance=float(appraisal.get("goal_relevance", 0.5)),
                    novelty=float(appraisal.get("novelty", 0.5)),
                    valence=float(appraisal.get("valence", 0.5)),
                    agency=float(appraisal.get("agency", 0.5)),
                    social_significance=float(
                        appraisal.get("social_significance", 0.5)
                    ),
                )
            except (ValueError, TypeError) as e:
                return {
                    "status": "error",
                    "error": "invalid_appraisal",
                    "message": str(e),
                }
        else:
            vector = rule_based_appraisal(event, source)

        # Run full appraisal
        result = run_appraisal(
            vector,
            sensitivity=sensitivity,
            resilience=resilience,
            base_half_life=config.episodes.base_half_life_minutes,
            formative_threshold=config.episodes.formative_intensity_threshold,
        )

        conv_id = uuid.UUID(conversation_id) if conversation_id else None

        # Create episode
        episode = create_episode(
            session,
            result,
            trigger_event=event,
            trigger_source=source,
            conversation_id=conv_id,
            event_bus=app.event_bus,
        )

        # Encode to memory immediately
        tags = [result.primary_emotion]
        if context:
            tags.append("contextual")
        mem = store_episodic_from_episode(
            session,
            app.embedding_service,
            episode=episode,
            content=event,
            conversation_id=conv_id,
            tags=tags,
            encoding_strength_weight=config.episodes.encoding_strength_weight,
            event_bus=app.event_bus,
        )

        session.commit()

        return {
            "status": "ok",
            "data": {
                "episode": {
                    "id": str(episode.id),
                    "appraisal": {
                        "goal_relevance": result.vector.goal_relevance,
                        "novelty": result.vector.novelty,
                        "valence": result.vector.valence,
                        "agency": result.vector.agency,
                        "social_significance": result.vector.social_significance,
                    },
                    "primary_emotion": result.primary_emotion,
                    "secondary_emotions": result.secondary_emotions,
                    "intensity": round(result.intensity, 4),
                    "half_life_minutes": round(result.half_life_minutes, 1),
                    "is_formative": result.is_formative,
                },
                "memory_id": str(mem.id),
                "mood_update": None,
                "note": "Mood updates available in Phase 2",
            },
        }

    except Exception as e:
        session.rollback()
        return {
            "status": "error",
            "error": "experience_event_failed",
            "message": str(e),
        }
    finally:
        session.close()
