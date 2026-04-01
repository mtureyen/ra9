"""Tests for the Mood subsystem (Phase 2)."""

from emotive.subsystems.mood.residue import EMOTION_TO_RESIDUE, MOOD_DIMENSIONS, compute_residue
from emotive.subsystems.mood.homeostasis import REUPTAKE_RATES, apply_homeostasis


class TestResidue:
    def test_all_emotions_have_residue(self):
        for emotion in ["joy", "sadness", "anger", "fear", "trust", "awe", "surprise", "disgust"]:
            assert emotion in EMOTION_TO_RESIDUE

    def test_neutral_has_no_residue(self):
        assert EMOTION_TO_RESIDUE["neutral"] == {}

    def test_compute_residue_scales_by_intensity(self):
        low = compute_residue("joy", 0.3)
        high = compute_residue("joy", 0.9)
        for dim in low:
            assert abs(high[dim]) > abs(low[dim])

    def test_joy_increases_social_bonding(self):
        residue = compute_residue("joy", 0.5)
        assert residue.get("social_bonding", 0) > 0

    def test_sadness_decreases_social_bonding(self):
        residue = compute_residue("sadness", 0.5)
        assert residue.get("social_bonding", 0) < 0

    def test_fear_increases_caution(self):
        residue = compute_residue("fear", 0.5)
        assert residue.get("caution", 0) > 0

    def test_trust_increases_social_decreases_caution(self):
        residue = compute_residue("trust", 0.5)
        assert residue.get("social_bonding", 0) > 0
        assert residue.get("caution", 0) < 0

    def test_neutral_returns_empty(self):
        residue = compute_residue("neutral", 0.5)
        assert residue == {}

    def test_unknown_emotion_returns_empty(self):
        residue = compute_residue("nonexistent", 0.5)
        assert residue == {}

    def test_zero_intensity_returns_zeros(self):
        residue = compute_residue("joy", 0.0)
        assert all(v == 0 for v in residue.values())


class TestHomeostasis:
    def test_all_dimensions_have_rates(self):
        for dim in MOOD_DIMENSIONS:
            assert dim in REUPTAKE_RATES

    def test_decays_toward_baseline(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        temp = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        result = apply_homeostasis(mood, temp, hours_elapsed=1.0)
        # novelty_seeking should move toward 0.5
        assert result["novelty_seeking"] < 0.8
        assert result["novelty_seeking"] > 0.5

    def test_no_change_at_baseline(self):
        mood = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        temp = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        result = apply_homeostasis(mood, temp, hours_elapsed=1.0)
        for dim in MOOD_DIMENSIONS:
            assert abs(result[dim] - 0.5) < 0.001

    def test_more_hours_more_decay(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        temp = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        short = apply_homeostasis(mood, temp, hours_elapsed=1.0)
        long = apply_homeostasis(mood, temp, hours_elapsed=10.0)
        assert long["novelty_seeking"] < short["novelty_seeking"]

    def test_novelty_decays_faster_than_social(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.8, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        temp = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        result = apply_homeostasis(mood, temp, hours_elapsed=1.0)
        # novelty rate 0.05 > social rate 0.02 → novelty decays more
        novelty_decay = 0.8 - result["novelty_seeking"]
        social_decay = 0.8 - result["social_bonding"]
        assert novelty_decay > social_decay

    def test_zero_hours_no_change(self):
        mood = {"novelty_seeking": 0.8, "social_bonding": 0.5, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        temp = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        result = apply_homeostasis(mood, temp, hours_elapsed=0.0)
        assert result["novelty_seeking"] == 0.8


class TestMoodSubsystem:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        assert mood.name == "mood"
        assert isinstance(mood.current, dict)

    def test_subscribes_to_episode_created(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        # Simulate episode creation
        event_bus.publish("episode_created", {
            "primary_emotion": "joy",
            "intensity": 0.7,
            "is_formative": False,
        })
        # Mood should have shifted
        assert mood.current["social_bonding"] > 0.5  # joy increases social bonding

    def test_neutral_episode_no_shift(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        before = dict(mood.current)
        event_bus.publish("episode_created", {
            "primary_emotion": "neutral",
            "intensity": 0.3,
        })
        assert mood.current == before

    def test_modulated_sensitivity_low_mood(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        # Push mood low with sad episodes
        for _ in range(5):
            event_bus.publish("episode_created", {
                "primary_emotion": "sadness",
                "intensity": 0.8,
            })
        # Low mood → sensitivity should increase
        modulated = mood.get_modulated_sensitivity(0.5)
        assert modulated > 0.5

    def test_modulated_sensitivity_high_mood(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        # Push mood high with joy episodes
        for _ in range(5):
            event_bus.publish("episode_created", {
                "primary_emotion": "joy",
                "intensity": 0.8,
            })
        # High mood → sensitivity should decrease
        modulated = mood.get_modulated_sensitivity(0.5)
        assert modulated < 0.5

    def test_publishes_mood_updated(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        events = []
        event_bus.subscribe("mood_updated", lambda t, d: events.append(d))

        mood = MoodSubsystem(app_context, event_bus)
        event_bus.publish("episode_created", {
            "primary_emotion": "trust",
            "intensity": 0.6,
        })
        assert len(events) == 1
        assert "previous" in events[0]
        assert "current" in events[0]
        assert "source_emotion" in events[0]

    def test_mood_clamped_0_to_1(self, app_context, event_bus):
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        # Push same dimension many times
        for _ in range(100):
            event_bus.publish("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.9,
            })
        # Should never exceed 1.0
        assert mood.current["caution"] <= 1.0
        assert mood.current["novelty_seeking"] >= 0.0


class TestMoodInContext:
    def test_mood_in_system_prompt(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        mood = {"novelty_seeking": 0.7, "social_bonding": 0.3, "analytical_depth": 0.5,
                "playfulness": 0.5, "caution": 0.5, "expressiveness": 0.5}
        prompt = build_system_prompt(mood=mood)
        assert "Mood" in prompt
        assert "novelty seeking" in prompt.lower()
        assert "social bonding" in prompt.lower()

    def test_baseline_mood_shows_neutral(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        mood = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        prompt = build_system_prompt(mood=mood)
        assert "neutral" in prompt.lower()

    def test_no_mood_no_section(self):
        from emotive.subsystems.prefrontal.context import build_system_prompt

        prompt = build_system_prompt(mood=None)
        assert "Mood" not in prompt
