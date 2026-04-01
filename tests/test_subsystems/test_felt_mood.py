"""Tests for mood natural language (felt descriptions, not numbers)."""

from emotive.subsystems.prefrontal.context import _format_mood, build_system_prompt


class TestFeltMoodDescriptions:
    def test_high_novelty_shows_curious(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        result = _format_mood(mood)
        assert "curious" in result.lower()
        # No raw numbers in output
        assert "0.8" not in result

    def test_low_social_shows_inward(self):
        mood = {"novelty_seeking": 0.5, "social_bonding": 0.2, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        result = _format_mood(mood)
        assert "inward" in result.lower()

    def test_high_caution_shows_guarded(self):
        mood = {"novelty_seeking": 0.5, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.85, "expressiveness": 0.5}
        result = _format_mood(mood)
        assert "guarded" in result.lower() or "careful" in result.lower()

    def test_low_playfulness_shows_serious(self):
        mood = {"novelty_seeking": 0.5, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.2, "caution": 0.5, "expressiveness": 0.5}
        result = _format_mood(mood)
        assert "serious" in result.lower()

    def test_high_expressiveness_shows_expressive(self):
        mood = {"novelty_seeking": 0.5, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.8}
        result = _format_mood(mood)
        assert "expressive" in result.lower() or "share" in result.lower()

    def test_baseline_shows_neutral(self):
        mood = {dim: 0.5 for dim in [
            "novelty_seeking", "social_bonding", "analytical_depth",
            "playfulness", "caution", "expressiveness",
        ]}
        result = _format_mood(mood)
        assert "neutral" in result.lower()

    def test_no_raw_numbers(self):
        mood = {"novelty_seeking": 0.7, "social_bonding": 0.3, "analytical_depth": 0.8,
                "playfulness": 0.2, "caution": 0.9, "expressiveness": 0.1}
        result = _format_mood(mood)
        # Should contain felt descriptions, not dimensional numbers
        assert "0.7" not in result
        assert "0.3" not in result
        assert "elevated" not in result  # old format word

    def test_slight_shift_uses_mild_qualifier(self):
        mood = {"novelty_seeking": 0.55, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        result = _format_mood(mood)
        assert "slightly" in result.lower()

    def test_multiple_dimensions_shifted(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.2, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.9, "expressiveness": 0.5}
        result = _format_mood(mood)
        # Should have multiple descriptions
        lines = [l for l in result.split("\n") if l.startswith("- ")]
        assert len(lines) == 3

    def test_in_system_prompt(self):
        mood = {"novelty_seeking": 0.7, "social_bonding": 0.3, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        prompt = build_system_prompt(mood=mood)
        assert "Mood" in prompt
        assert "curious" in prompt.lower()
