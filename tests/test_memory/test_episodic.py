"""Tests for episodic memory."""

import uuid

from emotive.memory.episodic import EPISODIC_DECAY_RATE, store_episodic


def test_store_episodic_sets_type(db_session, embedding_service):
    mem = store_episodic(db_session, embedding_service, content="episode test")
    assert mem.memory_type == "episodic"


def test_store_episodic_sets_decay_rate(db_session, embedding_service):
    mem = store_episodic(db_session, embedding_service, content="decay rate test")
    assert mem.decay_rate == EPISODIC_DECAY_RATE


def test_store_episodic_with_context(db_session, embedding_service):
    mem = store_episodic(
        db_session, embedding_service,
        content="context test", context={"participant": "user123"},
    )
    assert mem.metadata_["participant"] == "user123"


def test_store_episodic_with_tags(db_session, embedding_service):
    mem = store_episodic(
        db_session, embedding_service,
        content="tags test", tags=["important", "social"],
    )
    assert "important" in mem.tags
    assert "social" in mem.tags


def test_store_episodic_with_conversation_id(db_session, embedding_service):
    cid = uuid.uuid4()
    mem = store_episodic(
        db_session, embedding_service,
        content="conv test", conversation_id=cid,
    )
    assert mem.conversation_id == cid


def test_store_episodic_from_episode_sets_emotional_metadata(
    db_session, embedding_service,
):
    from emotive.db.models.episode import EmotionalEpisode
    from emotive.memory.episodic import store_episodic_from_episode

    # Create a fake episode
    ep = EmotionalEpisode(
        trigger_event="emotional test",
        trigger_source="test",
        appraisal_goal_relevance=0.8,
        appraisal_novelty=0.6,
        appraisal_valence=0.75,
        appraisal_agency=0.3,
        appraisal_social_significance=0.9,
        primary_emotion="trust",
        intensity=0.7,
        decay_rate=0.02,
        half_life_minutes=30.0,
    )
    db_session.add(ep)
    db_session.flush()

    mem = store_episodic_from_episode(
        db_session, embedding_service,
        episode=ep, content="emotional memory test",
    )
    assert mem.emotional_intensity == 0.7
    assert mem.primary_emotion == "trust"
    assert mem.valence == 0.75
    assert mem.source_episode_id == ep.id
    assert mem.decay_protection < 1.0  # stronger encoding = more protection
    assert ep.memory_encoded is True


def test_store_episodic_from_episode_encoding_strength(
    db_session, embedding_service,
):
    from emotive.db.models.episode import EmotionalEpisode
    from emotive.memory.episodic import store_episodic_from_episode

    # High intensity episode
    ep = EmotionalEpisode(
        trigger_event="intense test",
        trigger_source="test",
        appraisal_goal_relevance=0.9,
        appraisal_novelty=0.8,
        appraisal_valence=0.9,
        appraisal_agency=0.5,
        appraisal_social_significance=0.9,
        primary_emotion="awe",
        intensity=0.9,
        decay_rate=0.02,
        half_life_minutes=30.0,
    )
    db_session.add(ep)
    db_session.flush()

    mem = store_episodic_from_episode(
        db_session, embedding_service,
        episode=ep, content="intense memory",
    )
    # High intensity should get more protection (lower value = slower decay)
    assert mem.decay_protection < 0.7
