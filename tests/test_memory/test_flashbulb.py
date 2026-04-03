"""Tests for flashbulb memories — near-permanent encoding for high-intensity formative events."""

from emotive.db.models.episode import EmotionalEpisode
from emotive.memory.episodic import store_episodic_from_episode


def _make_episode(db_session, emotion, intensity, is_formative, valence=0.5):
    """Create and flush an episode for testing."""
    ep = EmotionalEpisode(
        trigger_event=f"Test event: {emotion} at {intensity}",
        trigger_source="test",
        primary_emotion=emotion,
        intensity=intensity,
        is_formative=is_formative,
        appraisal_valence=valence,
        appraisal_goal_relevance=0.5,
        appraisal_novelty=0.5,
        appraisal_agency=0.5,
        appraisal_social_significance=0.5,
        decay_rate=0.023,
        half_life_minutes=30.0,
    )
    db_session.add(ep)
    db_session.flush()
    return ep


class TestFlashbulbMemory:
    def test_formative_high_intensity_gets_flashbulb_protection(
        self, db_session, embedding_service
    ):
        """Formative events with intensity > 0.8 get decay_protection = 0.1."""
        ep = _make_episode(db_session, "awe", 0.9, True, 0.8)
        mem = store_episodic_from_episode(
            db_session, embedding_service,
            episode=ep,
            content="The moment I chose my name — ryo",
            tags=["identity", "naming"],
        )
        assert mem.decay_protection == 0.1

    def test_formative_low_intensity_normal_protection(
        self, db_session, embedding_service
    ):
        """Formative events below 0.8 intensity get normal protection."""
        ep = _make_episode(db_session, "trust", 0.6, True, 0.7)
        mem = store_episodic_from_episode(
            db_session, embedding_service,
            episode=ep,
            content="A meaningful trust moment",
            tags=["trust"],
        )
        assert mem.decay_protection > 0.1

    def test_non_formative_high_intensity_normal_protection(
        self, db_session, embedding_service
    ):
        """Non-formative high intensity events don't get flashbulb."""
        ep = _make_episode(db_session, "anger", 0.85, False, 0.2)
        mem = store_episodic_from_episode(
            db_session, embedding_service,
            episode=ep,
            content="An angry moment",
            tags=["anger"],
        )
        assert mem.decay_protection > 0.1

    def test_flashbulb_is_slowest_decay(self, db_session, embedding_service):
        """Flashbulb memories should have the slowest decay of any episodic."""
        ep = _make_episode(db_session, "joy", 0.95, True, 0.9)
        mem = store_episodic_from_episode(
            db_session, embedding_service,
            episode=ep,
            content="The most important moment",
            tags=["identity"],
        )
        assert mem.decay_protection == 0.1
        assert mem.decay_protection < 0.5
