"""Tests for the Self-Output Appraisal subsystem (Phase 2.5 G)."""

import pytest

from emotive.subsystems.acc.tone_monitor import (
    NUDGE_KEYWORDS,
    check_tone_alignment,
)
from emotive.subsystems.dmn.discovery import detect_discovery
from emotive.subsystems.acc.self_appraisal import SelfAppraisal


# ---------------------------------------------------------------------------
# Pure function tests: check_tone_alignment
# ---------------------------------------------------------------------------

class TestCheckToneAlignment:
    def test_warm_keywords_match(self):
        response = "I really appreciate you sharing that, I'm glad we talked"
        alignment = check_tone_alignment(response, "warm")
        assert alignment > 0.0

    def test_no_keywords_returns_neutral(self):
        """Nudges without keywords get 0.5 (neutral)."""
        alignment = check_tone_alignment("Hello there!", "present")
        assert alignment == 0.5

    def test_unknown_nudge_returns_neutral(self):
        alignment = check_tone_alignment("Hello", "some_random_nudge")
        assert alignment == 0.5

    def test_guard_keywords_match(self):
        response = "I'm not sure about that, I won't be able to help with that"
        alignment = check_tone_alignment(response, "guard")
        assert alignment > 0.0

    def test_no_match_returns_zero(self):
        response = "The weather is nice today"
        alignment = check_tone_alignment(response, "warm")
        assert alignment == 0.0

    def test_all_keywords_match_capped_at_one(self):
        response = "I care and appreciate and am glad and thank and love you"
        alignment = check_tone_alignment(response, "warm")
        assert alignment == 1.0

    def test_case_insensitive(self):
        response = "I CARE about this deeply"
        alignment = check_tone_alignment(response, "warm")
        assert alignment > 0.0


# ---------------------------------------------------------------------------
# Pure function tests: detect_discovery
# ---------------------------------------------------------------------------

class TestDetectDiscovery:
    def test_no_recalled_is_discovery(self):
        """If nothing was recalled, everything is new."""
        assert detect_discovery([0.1] * 4, []) is True

    def test_similar_response_not_discovery(self):
        """Response very close to a recalled memory is not discovery."""
        vec = [1.0, 0.0, 0.0, 0.0]
        assert detect_discovery(vec, [vec]) is False

    def test_distant_response_is_discovery(self):
        """Response far from all recalled memories is discovery."""
        response = [1.0, 0.0, 0.0, 0.0]
        recalled = [[0.0, 0.0, 0.0, 1.0]]
        assert detect_discovery(response, recalled) is True


# ---------------------------------------------------------------------------
# Subsystem tests
# ---------------------------------------------------------------------------

class TestSelfAppraisal:
    def test_evaluate_returns_dict(self, app_context, event_bus):
        sa = SelfAppraisal(app_context, event_bus)
        result = sa.evaluate(
            response_text="I appreciate you sharing that",
            response_embedding=[0.1] * 4,
            nudge="warm",
            inner_speech=None,
            recalled_embeddings=[[0.1] * 4],
        )
        assert "tone_alignment" in result
        assert "discovery" in result

    def test_publishes_tone_misalignment(self, app_context, event_bus):
        from emotive.runtime.event_bus import TONE_MISALIGNMENT
        received = []
        event_bus.subscribe(TONE_MISALIGNMENT, lambda et, d: received.append(d))

        sa = SelfAppraisal(app_context, event_bus)
        sa.evaluate(
            response_text="The weather is nice",
            response_embedding=[0.1] * 4,
            nudge="warm",  # No warm keywords in response
            inner_speech=None,
            recalled_embeddings=[[0.1] * 4],
        )
        assert len(received) == 1
        assert received[0]["nudge"] == "warm"
        assert received[0]["alignment"] == 0.0

    def test_publishes_discovery_detected(self, app_context, event_bus):
        from emotive.runtime.event_bus import DISCOVERY_DETECTED
        received = []
        event_bus.subscribe(DISCOVERY_DETECTED, lambda et, d: received.append(d))

        sa = SelfAppraisal(app_context, event_bus)
        sa.evaluate(
            response_text="Something entirely new",
            response_embedding=[1.0, 0.0, 0.0, 0.0],
            nudge="present",
            inner_speech=None,
            recalled_embeddings=[[0.0, 0.0, 0.0, 1.0]],
        )
        assert len(received) == 1

    def test_no_misalignment_event_when_aligned(self, app_context, event_bus):
        from emotive.runtime.event_bus import TONE_MISALIGNMENT
        received = []
        event_bus.subscribe(TONE_MISALIGNMENT, lambda et, d: received.append(d))

        sa = SelfAppraisal(app_context, event_bus)
        sa.evaluate(
            response_text="I care and appreciate and am glad",
            response_embedding=[0.1] * 4,
            nudge="warm",
            inner_speech=None,
            recalled_embeddings=[[0.1] * 4],
        )
        # alignment > 0.3, so no event
        assert len(received) == 0

    def test_publishes_self_appraisal_complete(self, app_context, event_bus):
        from emotive.runtime.event_bus import SELF_APPRAISAL_COMPLETE
        received = []
        event_bus.subscribe(SELF_APPRAISAL_COMPLETE, lambda et, d: received.append(d))

        sa = SelfAppraisal(app_context, event_bus)
        sa.evaluate(
            response_text="Hello",
            response_embedding=[0.1] * 4,
            nudge="present",
            inner_speech=None,
            recalled_embeddings=[],
        )
        assert len(received) == 1
        assert "tone_alignment" in received[0]
        assert "discovery" in received[0]
