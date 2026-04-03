"""Tests for Phase 2.5 inner world additions to PFC context builder."""

import pytest

from emotive.subsystems.prefrontal.context import (
    build_system_prompt,
    _format_embodied_state,
    _format_inner_world,
    _format_social_perception,
)


class TestBuildSystemPromptBackwardCompat:
    """Existing call sites must not break with new optional kwargs."""

    def test_build_without_inner_world_kwargs(self):
        """Original signature still works with no inner world args."""
        prompt = build_system_prompt()
        assert "Ryo" in prompt
        assert "Inner Voice" not in prompt

    def test_build_with_all_none_inner_world_kwargs(self):
        """Explicit None values don't add sections."""
        prompt = build_system_prompt(
            inner_voice_nudge=None,
            inner_speech=None,
            embodied_state=None,
            social_perception=None,
            metacognitive_markers=None,
        )
        assert "Inner Voice" not in prompt
        assert "How You Feel" not in prompt
        assert "Reading the Room" not in prompt
        assert "Self-Awareness" not in prompt


class TestFormatInnerWorld:
    """Test _format_inner_world function."""

    def test_nudge_only(self):
        result = _format_inner_world("warmth", None)
        assert "Your Inner Voice" in result
        assert "You're feeling: warmth" in result
        assert "Private thought" not in result

    def test_inner_speech_only(self):
        result = _format_inner_world(None, "I should be careful here")
        assert "Your Inner Voice" in result
        assert "Private thought: I should be careful here" in result
        assert "You're feeling" not in result

    def test_both_nudge_and_speech(self):
        result = _format_inner_world("caution", "Something feels off")
        assert "You're feeling: caution" in result
        assert "Private thought: Something feels off" in result

    def test_both_none_returns_header_only(self):
        result = _format_inner_world(None, None)
        assert "Your Inner Voice" in result


class TestFormatEmbodiedState:
    """Test _format_embodied_state function."""

    def test_low_energy(self):
        result = _format_embodied_state({"energy": 0.2, "comfort": 0.5, "cognitive_load": 0.0})
        assert "tired" in result
        assert "How You Feel" in result

    def test_moderate_energy(self):
        result = _format_embodied_state({"energy": 0.4, "comfort": 0.5, "cognitive_load": 0.0})
        assert "moderate" in result

    def test_high_comfort(self):
        result = _format_embodied_state({"energy": 1.0, "comfort": 0.8, "cognitive_load": 0.0})
        assert "comfortable" in result
        assert "at ease" in result

    def test_low_comfort(self):
        result = _format_embodied_state({"energy": 1.0, "comfort": 0.2, "cognitive_load": 0.0})
        assert "guarded" in result

    def test_high_cognitive_load(self):
        result = _format_embodied_state({"energy": 1.0, "comfort": 0.5, "cognitive_load": 0.8})
        assert "complex" in result

    def test_all_baseline_returns_empty(self):
        """All values at neutral should produce empty string."""
        result = _format_embodied_state({"energy": 1.0, "comfort": 0.5, "cognitive_load": 0.0})
        assert result == ""

    def test_defaults_when_keys_missing(self):
        """Missing keys should use defaults and not crash."""
        result = _format_embodied_state({})
        assert result == ""


class TestFormatSocialPerception:
    """Test _format_social_perception function."""

    def test_with_user_state(self):
        result = _format_social_perception("curious")
        assert "Reading the Room" in result
        assert "curious" in result

    def test_none_returns_empty(self):
        result = _format_social_perception(None)
        assert result == ""

    def test_empty_string_returns_empty(self):
        result = _format_social_perception("")
        assert result == ""


class TestBuildSystemPromptWithInnerWorld:
    """Test full prompt assembly with inner world sections."""

    def test_inner_voice_nudge_in_prompt(self):
        prompt = build_system_prompt(inner_voice_nudge="warmth")
        assert "Your Inner Voice" in prompt
        assert "warmth" in prompt

    def test_inner_speech_in_prompt(self):
        prompt = build_system_prompt(inner_speech="I should be gentle here")
        assert "Private thought" in prompt

    def test_embodied_state_in_prompt(self):
        prompt = build_system_prompt(
            embodied_state={"energy": 0.2, "comfort": 0.8, "cognitive_load": 0.0}
        )
        assert "How You Feel" in prompt
        assert "tired" in prompt

    def test_social_perception_in_prompt(self):
        prompt = build_system_prompt(social_perception="upset")
        assert "Reading the Room" in prompt
        assert "upset" in prompt

    def test_metacognitive_markers_in_prompt(self):
        prompt = build_system_prompt(
            metacognitive_markers="I'm not sure I remember this well"
        )
        assert "Self-Awareness" in prompt
        assert "not sure" in prompt

    def test_inner_world_sections_order(self):
        """Inner world sections should appear after mood, before emotional state."""
        from emotive.layers.appraisal import AppraisalResult, AppraisalVector

        vector = AppraisalVector(
            valence=0.5, novelty=0.5, goal_relevance=0.5,
            agency=0.5, social_significance=0.5,
        )
        appraisal = AppraisalResult(
            vector=vector,
            primary_emotion="joy",
            secondary_emotions=[],
            intensity=0.6,
            half_life_minutes=30.0,
            is_formative=False,
            decay_rate=0.023,
        )

        prompt = build_system_prompt(
            mood={"novelty_seeking": 0.8},
            emotional_state=appraisal,
            inner_voice_nudge="warmth",
            embodied_state={"energy": 0.2, "comfort": 0.5, "cognitive_load": 0.0},
        )

        mood_pos = prompt.find("Your Current Mood")
        inner_voice_pos = prompt.find("Your Inner Voice")
        embodied_pos = prompt.find("How You Feel")
        emotion_pos = prompt.find("Current Emotional State")

        assert mood_pos < inner_voice_pos < embodied_pos < emotion_pos
