"""Tests for the Inner Voice subsystem (Phase 2.5 E)."""

import pytest

from emotive.subsystems.inner_voice.rules import compute_nudge
from emotive.subsystems.inner_voice import InnerVoice
from emotive.subsystems.prefrontal.metacognition.markers import MetacognitiveMarkers


def _default_metacog(**kwargs):
    defaults = {"memory_confidence": 0.8, "emotional_clarity": 0.8, "knowledge_confidence": 0.8}
    defaults.update(kwargs)
    return MetacognitiveMarkers(**defaults)


def _default_mood(**kwargs):
    mood = {"social_bonding": 0.5, "caution": 0.5, "novelty_seeking": 0.5, "playfulness": 0.5}
    mood.update(kwargs)
    return mood


# ---------------------------------------------------------------------------
# Pure function tests: compute_nudge
# ---------------------------------------------------------------------------

class TestComputeNudge:
    def test_exhausted_overrides_all(self):
        assert compute_nudge(_default_mood(), "core", "playful", _default_metacog(), energy=0.1, comfort=0.9) == "exhausted"

    def test_unknown_confrontational_guard(self):
        assert compute_nudge(_default_mood(), "unknown", "confrontational", _default_metacog(), energy=0.5, comfort=0.5) == "guard"

    def test_unknown_user_cautious(self):
        assert compute_nudge(_default_mood(), "unknown", None, _default_metacog(), energy=0.5, comfort=0.5) == "cautious"

    def test_low_knowledge_uncertain(self):
        metacog = _default_metacog(knowledge_confidence=0.2)
        assert compute_nudge(_default_mood(), "trusted", None, metacog, energy=0.5, comfort=0.5) == "uncertain"

    def test_high_caution_watchful(self):
        mood = _default_mood(caution=0.9)
        assert compute_nudge(mood, "trusted", None, _default_metacog(), energy=0.5, comfort=0.5) == "watchful"

    def test_comfortable_playful_user(self):
        assert compute_nudge(_default_mood(), "trusted", "playful", _default_metacog(), energy=0.5, comfort=0.8) == "playful"

    def test_warm_when_bonded_comfortable(self):
        mood = _default_mood(social_bonding=0.7)
        assert compute_nudge(mood, "trusted", None, _default_metacog(), energy=0.5, comfort=0.7) == "warm"

    def test_gentle_with_vulnerable_user(self):
        assert compute_nudge(_default_mood(), "trusted", "vulnerable", _default_metacog(), energy=0.5, comfort=0.5) == "gentle"

    def test_careful_with_upset_user(self):
        assert compute_nudge(_default_mood(), "trusted", "upset", _default_metacog(), energy=0.5, comfort=0.5) == "careful"

    def test_curious_when_novelty_seeking(self):
        mood = _default_mood(novelty_seeking=0.7)
        assert compute_nudge(mood, "trusted", "curious", _default_metacog(), energy=0.5, comfort=0.5) == "curious"

    def test_default_present(self):
        assert compute_nudge(_default_mood(), "trusted", None, _default_metacog(), energy=0.5, comfort=0.5) == "present"

    def test_priority_exhausted_over_guard(self):
        """Exhaustion takes priority over everything else."""
        assert compute_nudge(_default_mood(), "unknown", "confrontational", _default_metacog(), energy=0.1, comfort=0.9) == "exhausted"


# ---------------------------------------------------------------------------
# Subsystem tests
# ---------------------------------------------------------------------------

class TestInnerVoiceSubsystem:
    def test_nudge_returns_string(self, app_context, event_bus):
        iv = InnerVoice(app_context, event_bus)
        result = iv.nudge(
            mood=_default_mood(),
            trust_level="trusted",
            user_state=None,
            metacog=_default_metacog(),
            energy=0.5,
            comfort=0.5,
        )
        assert isinstance(result, str)
        assert result == "present"

    def test_publishes_nudge_event(self, app_context, event_bus):
        from emotive.runtime.event_bus import INNER_VOICE_NUDGE
        received = []
        event_bus.subscribe(INNER_VOICE_NUDGE, lambda et, d: received.append(d))

        iv = InnerVoice(app_context, event_bus)
        iv.nudge(
            mood=_default_mood(),
            trust_level="unknown",
            user_state=None,
            metacog=_default_metacog(),
            energy=0.5,
            comfort=0.5,
        )
        assert len(received) == 1
        assert received[0]["nudge"] == "cautious"
