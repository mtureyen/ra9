"""Tests for episode manager."""

from datetime import datetime, timedelta, timezone

from emotive.db.models.episode import EmotionalEpisode
from emotive.layers.appraisal import AppraisalVector, run_appraisal
from emotive.layers.episodes import (
    archive_decayed_episodes,
    create_episode,
    get_active_episodes,
    get_current_intensity,
    get_unencoded_episodes,
)


def _make_appraisal(**kwargs):
    defaults = {
        "goal_relevance": 0.7, "novelty": 0.5, "valence": 0.8,
        "agency": 0.3, "social_significance": 0.6,
    }
    defaults.update(kwargs)
    v = AppraisalVector(**defaults)
    return run_appraisal(v)


def test_create_episode_persists(db_session):
    result = _make_appraisal()
    ep = create_episode(
        db_session, result, trigger_event="test event", trigger_source="test",
    )
    db_session.flush()
    assert ep.id is not None
    assert ep.primary_emotion == result.primary_emotion
    assert ep.intensity == result.intensity


def test_create_episode_with_conversation(db_session):
    import uuid

    result = _make_appraisal()
    conv_id = uuid.uuid4()
    ep = create_episode(
        db_session, result,
        trigger_event="conv test", trigger_source="test",
        conversation_id=conv_id,
    )
    assert ep.conversation_id == conv_id


def test_create_episode_publishes_event(db_session, event_bus):
    received = []
    event_bus.subscribe("episode_created", lambda t, d: received.append(d))
    result = _make_appraisal()
    create_episode(
        db_session, result,
        trigger_event="event bus test", trigger_source="test",
        event_bus=event_bus,
    )
    assert len(received) == 1
    assert received[0]["primary_emotion"] == result.primary_emotion


def test_create_episode_formative_flag(db_session):
    result = _make_appraisal(
        goal_relevance=0.95, novelty=0.9, valence=0.95,
        agency=0.8, social_significance=0.95,
    )
    # Use high sensitivity to push intensity above formative threshold
    result = run_appraisal(
        result.vector, sensitivity=0.8, formative_threshold=0.8,
    )
    ep = create_episode(
        db_session, result, trigger_event="formative test", trigger_source="test",
    )
    assert ep.is_formative is True


def test_get_active_episodes(db_session):
    result = _make_appraisal()
    create_episode(db_session, result, trigger_event="active 1", trigger_source="test")
    create_episode(db_session, result, trigger_event="active 2", trigger_source="test")
    db_session.flush()

    active = get_active_episodes(db_session)
    assert len(active) >= 2


def test_get_current_intensity_decays():
    ep = EmotionalEpisode(
        trigger_event="test",
        appraisal_goal_relevance=0.5, appraisal_novelty=0.5,
        appraisal_valence=0.5, appraisal_agency=0.5,
        appraisal_social_significance=0.5,
        primary_emotion="joy", intensity=0.8,
        decay_rate=0.023, half_life_minutes=30.0,
        created_at=datetime.now(timezone.utc) - timedelta(minutes=60),
    )
    current = get_current_intensity(ep)
    # After 60 min with 30 min half-life, should be ~0.2
    assert current < ep.intensity
    assert current > 0


def test_archive_decayed_episodes(db_session):
    result = _make_appraisal()
    ep = create_episode(
        db_session, result, trigger_event="archive test", trigger_source="test",
    )
    db_session.flush()

    # Manually set created_at far in the past so it's decayed
    ep.created_at = datetime.now(timezone.utc) - timedelta(hours=24)
    ep.half_life_minutes = 1.0  # 1 minute half-life
    db_session.flush()

    archived = archive_decayed_episodes(db_session, sensitivity=0.5)
    assert archived >= 1
    db_session.refresh(ep)
    assert ep.is_active is False


def test_get_unencoded_episodes(db_session):
    result = _make_appraisal()
    ep = create_episode(
        db_session, result, trigger_event="unencoded test", trigger_source="test",
    )
    db_session.flush()
    assert ep.memory_encoded is False

    unencoded = get_unencoded_episodes(db_session)
    assert any(e.id == ep.id for e in unencoded)
