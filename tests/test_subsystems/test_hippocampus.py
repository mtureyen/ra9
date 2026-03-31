"""Tests for the Hippocampus subsystem (encoding + intent detection)."""

import time

import pytest

from emotive.config.schema import UnconsciousEncodingConfig
from emotive.layers.appraisal import AppraisalResult, AppraisalVector
from emotive.subsystems.hippocampus.encoding import UnconsciousEncoder
from emotive.subsystems.hippocampus.intent import (
    detect_encoding_intent,
    enhanced_encode,
)


def _make_appraisal(intensity: float, emotion: str = "joy") -> AppraisalResult:
    return AppraisalResult(
        vector=AppraisalVector(
            goal_relevance=0.7, novelty=0.5, valence=0.8,
            agency=0.5, social_significance=0.6,
        ),
        primary_emotion=emotion,
        secondary_emotions=[],
        intensity=intensity,
        half_life_minutes=30.0,
        is_formative=intensity > 0.8,
        decay_rate=0.023,
    )


class TestUnconsciousEncoder:
    def test_below_threshold_does_not_encode(self):
        encoder = UnconsciousEncoder(UnconsciousEncodingConfig(intensity_threshold=0.5))
        assert encoder.should_encode(0.3) is False

    def test_above_threshold_encodes(self):
        encoder = UnconsciousEncoder(UnconsciousEncodingConfig(intensity_threshold=0.3))
        assert encoder.should_encode(0.5) is True

    def test_max_per_exchange(self):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(
                intensity_threshold=0.1, max_per_exchange=2, cooldown_seconds=0
            )
        )
        assert encoder.should_encode(0.5) is True
        encoder._exchange_count = 2
        assert encoder.should_encode(0.5) is False

    def test_reset_exchange(self):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.1, max_per_exchange=1)
        )
        encoder._exchange_count = 1
        assert encoder.should_encode(0.5) is False
        encoder.reset_exchange()
        assert encoder.should_encode(0.5) is True

    def test_cooldown(self):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(
                intensity_threshold=0.1, cooldown_seconds=100, max_per_exchange=10
            )
        )
        encoder._last_encode_time = time.monotonic()
        assert encoder.should_encode(0.5) is False

    def test_cooldown_expired(self):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(
                intensity_threshold=0.1, cooldown_seconds=0, max_per_exchange=10
            )
        )
        assert encoder.should_encode(0.5) is True

    def test_encode_creates_episode_and_memory(
        self, db_session, embedding_service, event_bus
    ):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(
                intensity_threshold=0.1, cooldown_seconds=0, max_per_exchange=3
            )
        )
        appraisal = _make_appraisal(0.6)
        memory, episode_id = encoder.encode(
            db_session, embedding_service, appraisal,
            "Something important happened",
            event_bus=event_bus,
        )
        assert memory is not None
        assert episode_id is not None
        assert memory.primary_emotion == "joy"
        assert memory.emotional_intensity == 0.6

    def test_encode_returns_none_below_threshold(
        self, db_session, embedding_service
    ):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.8)
        )
        appraisal = _make_appraisal(0.3)
        memory, episode_id = encoder.encode(
            db_session, embedding_service, appraisal,
            "Not important enough",
        )
        assert memory is None
        assert episode_id is None

    def test_encode_increments_counters(self, db_session, embedding_service):
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(
                intensity_threshold=0.1, cooldown_seconds=0, max_per_exchange=3
            )
        )
        assert encoder._exchange_count == 0
        appraisal = _make_appraisal(0.6)
        encoder.encode(
            db_session, embedding_service, appraisal, "test content"
        )
        assert encoder._exchange_count == 1


class TestIntentDetection:
    def test_detects_remember(self):
        assert detect_encoding_intent("I want to remember this moment")

    def test_detects_note_to_self(self):
        assert detect_encoding_intent("Note to self: check this later")

    def test_detects_important(self):
        assert detect_encoding_intent("This is important to me")

    def test_detects_wont_forget(self):
        assert detect_encoding_intent("I won't forget what you said")

    def test_detects_dont_let_me_forget(self):
        assert detect_encoding_intent("Don't let me forget about this")

    def test_no_false_positive_normal_text(self):
        assert not detect_encoding_intent("The weather is nice today")

    def test_no_false_positive_memory_discussion(self):
        assert not detect_encoding_intent("How does memory work in the brain?")

    def test_case_insensitive(self):
        assert detect_encoding_intent("I WANT TO REMEMBER THIS")

    def test_enhanced_encode_creates_memory(self, db_session, embedding_service):
        enhanced_encode(
            db_session, embedding_service,
            "This is something I want to keep",
        )
        # Should not raise
        db_session.flush()


class TestHippocampusSubsystem:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        assert hippo.name == "hippocampus"

    def test_process_appraisal_below_threshold(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        appraisal = _make_appraisal(0.1)  # Below default 0.4 threshold
        memory, episode_id = hippo.process_appraisal(
            appraisal, "low intensity message", "response", None
        )
        assert memory is None
        assert episode_id is None

    def test_detect_intent_positive(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        result = hippo.detect_intent("I want to remember this forever", None)
        assert result is True

    def test_detect_intent_negative(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        result = hippo.detect_intent("Just a normal response", None)
        assert result is False

    def test_store_gist(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        stored_events = []
        gist_events = []
        event_bus.subscribe("memory_stored", lambda t, d: stored_events.append(d))
        event_bus.subscribe("gist_created", lambda t, d: gist_events.append(d))

        hippo = Hippocampus(app_context, event_bus)
        memory = hippo.store_gist("We discussed the architecture", None)
        assert memory is not None

        # Verify via events (memory object may be detached from session)
        assert len(gist_events) == 1
        assert "content" in gist_events[0]

        # memory_stored event confirms correct type
        gist_stored = [
            e for e in stored_events
            if "architecture" in e.get("content", "")
        ]
        assert len(gist_stored) == 1
        assert gist_stored[0]["memory_type"] == "episodic"

    def test_reset_exchange(self, app_context, event_bus):
        from emotive.subsystems.hippocampus import Hippocampus

        hippo = Hippocampus(app_context, event_bus)
        hippo._encoder._exchange_count = 3
        hippo.reset_exchange()
        assert hippo._encoder._exchange_count == 0
