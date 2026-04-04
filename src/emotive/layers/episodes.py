"""Episode manager — create, decay, archive emotional episodes."""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.db.models.episode import EmotionalEpisode
from emotive.layers.appraisal import AppraisalResult
from emotive.runtime.event_bus import EPISODE_CREATED, EventBus

EPISODE_ARCHIVED = "episode_archived"


def create_episode(
    session: Session,
    appraisal: AppraisalResult,
    *,
    trigger_event: str,
    trigger_source: str = "user_message",
    conversation_id: uuid.UUID | None = None,
    event_bus: EventBus | None = None,
) -> EmotionalEpisode:
    """Create an emotional episode from an appraisal result."""
    v = appraisal.vector

    # Compute mood deltas from emotion + intensity
    from emotive.subsystems.raphe.residue import compute_residue

    mood_deltas = compute_residue(appraisal.primary_emotion, appraisal.intensity)

    episode = EmotionalEpisode(
        trigger_event=trigger_event,
        trigger_source=trigger_source,
        conversation_id=conversation_id,
        appraisal_goal_relevance=v.goal_relevance,
        appraisal_novelty=v.novelty,
        appraisal_valence=v.valence,
        appraisal_agency=v.agency,
        appraisal_social_significance=v.social_significance,
        primary_emotion=appraisal.primary_emotion,
        secondary_emotions=appraisal.secondary_emotions or [],
        intensity=appraisal.intensity,
        decay_rate=appraisal.decay_rate,
        half_life_minutes=appraisal.half_life_minutes,
        is_formative=appraisal.is_formative,
        # Mood dimensional deltas (Phase 2+)
        delta_novelty_seeking=mood_deltas.get("novelty_seeking", 0),
        delta_social_bonding=mood_deltas.get("social_bonding", 0),
        delta_analytical_depth=mood_deltas.get("analytical_depth", 0),
        delta_playfulness=mood_deltas.get("playfulness", 0),
        delta_caution=mood_deltas.get("caution", 0),
        delta_expressiveness=mood_deltas.get("expressiveness", 0),
    )

    session.add(episode)
    session.flush()

    if event_bus:
        event_bus.publish(
            EPISODE_CREATED,
            {
                "primary_emotion": appraisal.primary_emotion,
                "intensity": appraisal.intensity,
                "is_formative": appraisal.is_formative,
                "trigger_event": trigger_event[:200],
            },
            episode_id=episode.id,
            conversation_id=conversation_id,
        )

    return episode


def get_active_episodes(session: Session) -> list[EmotionalEpisode]:
    """Return all active (non-archived) episodes, newest first."""
    stmt = (
        select(EmotionalEpisode)
        .where(EmotionalEpisode.is_active.is_(True))
        .order_by(EmotionalEpisode.created_at.desc())
    )
    return list(session.execute(stmt).scalars().all())


def get_current_intensity(episode: EmotionalEpisode) -> float:
    """Calculate the current decayed intensity of an episode."""
    now = datetime.now(timezone.utc)
    created = episode.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    elapsed_minutes = (now - created).total_seconds() / 60.0
    if episode.half_life_minutes <= 0:
        return 0.0

    return episode.intensity * math.pow(
        0.5, elapsed_minutes / episode.half_life_minutes
    )


def archive_decayed_episodes(
    session: Session,
    sensitivity: float = 0.5,
    *,
    event_bus: EventBus | None = None,
) -> int:
    """Archive episodes whose current intensity has decayed below threshold.

    The threshold is the temperament sensitivity — episodes below this
    level are no longer emotionally significant enough to remain active.
    """
    active = get_active_episodes(session)
    archived_count = 0
    now = datetime.now(timezone.utc)

    for ep in active:
        current = get_current_intensity(ep)
        if current < sensitivity * 0.1:  # well below sensitivity threshold
            ep.is_active = False
            ep.archived_at = now
            archived_count += 1

            if event_bus:
                event_bus.publish(
                    EPISODE_ARCHIVED,
                    {
                        "primary_emotion": ep.primary_emotion,
                        "final_intensity": current,
                    },
                    episode_id=ep.id,
                )

    if archived_count > 0:
        session.flush()

    return archived_count


def get_unencoded_episodes(session: Session) -> list[EmotionalEpisode]:
    """Return episodes that haven't been encoded into memory yet."""
    stmt = (
        select(EmotionalEpisode)
        .where(EmotionalEpisode.memory_encoded.is_(False))
        .order_by(EmotionalEpisode.created_at)
    )
    return list(session.execute(stmt).scalars().all())
