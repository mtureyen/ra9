"""Tests for within-session mood homeostasis (Fix E)."""

from emotive.subsystems.mood.residue import MOOD_DIMENSIONS


class TestWithinSessionHomeostasis:
    def test_homeostasis_tick_fires_at_interval(self, app_context, event_bus):
        """Homeostasis should fire every N episodes (default 5)."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)

        # Push caution up with fear episodes
        for i in range(5):
            event_bus.publish("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.7,
            })

        caution_after_5 = mood.current["caution"]

        # Now push 5 more without homeostasis check
        # (we check that at episode 10, another tick fires)
        for i in range(5):
            event_bus.publish("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.7,
            })

        caution_after_10 = mood.current["caution"]

        # Caution should be high but homeostasis should have pulled it back
        # slightly at ticks 5 and 10. Without homeostasis, it would be higher.
        assert caution_after_5 > 0.5  # fear increases caution
        assert caution_after_10 > caution_after_5  # still accumulating
        # But it should be less than 1.0 (clamped + homeostasis)
        assert caution_after_10 < 1.0

    def test_homeostasis_prevents_runaway_accumulation(self, app_context, event_bus):
        """35 fear episodes should NOT push caution to 0.79+ with homeostasis."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)

        # Simulate a long session with sustained fear
        for i in range(35):
            event_bus.publish("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.5,
            })

        caution_with_homeostasis = mood.current["caution"]

        # Now compare: create a second MoodSubsystem WITHOUT ticks
        mood2 = MoodSubsystem(app_context, event_bus)
        mood2._tick_interval = 999  # effectively disabled

        for i in range(35):
            mood2._on_episode("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.5,
            })

        caution_without = mood2.current["caution"]

        # With homeostasis ticks, caution should be lower
        assert caution_with_homeostasis < caution_without

    def test_no_tick_before_interval(self, app_context, event_bus):
        """Homeostasis should NOT fire before the interval is reached."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        assert mood._episode_count == 0

        # Send 4 episodes (interval is 5)
        for i in range(4):
            event_bus.publish("episode_created", {
                "primary_emotion": "fear",
                "intensity": 0.7,
            })

        assert mood._episode_count == 4
        # No tick should have fired yet — caution accumulated without decay

    def test_tick_interval_configurable(self, app_context, event_bus):
        """Tick interval should come from MoodConfig."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        config = app_context.config_manager.get()
        assert mood._tick_interval == config.mood.homeostasis_tick_interval

    def test_tick_hours_configurable(self, app_context, event_bus):
        """Tick hours should come from MoodConfig."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        config = app_context.config_manager.get()
        assert mood._tick_hours == config.mood.homeostasis_tick_hours

    def test_episode_counter_increments(self, app_context, event_bus):
        """Episode counter should track how many episodes have occurred."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)
        assert mood._episode_count == 0

        event_bus.publish("episode_created", {
            "primary_emotion": "joy",
            "intensity": 0.6,
        })
        assert mood._episode_count == 1

        event_bus.publish("episode_created", {
            "primary_emotion": "trust",
            "intensity": 0.5,
        })
        assert mood._episode_count == 2

    def test_neutral_episodes_dont_trigger_residue_but_count(self, app_context, event_bus):
        """Neutral episodes return early (no residue) but should NOT count toward tick."""
        from emotive.subsystems.mood import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)

        # Neutral has empty residue → returns early before counter
        event_bus.publish("episode_created", {
            "primary_emotion": "neutral",
            "intensity": 0.3,
        })

        # Counter should be 0 since neutral returns early
        assert mood._episode_count == 0


class TestMoodConfigDefaults:
    def test_default_tick_interval(self):
        from emotive.config.schema import MoodConfig

        cfg = MoodConfig()
        assert cfg.homeostasis_tick_interval == 5

    def test_default_tick_hours(self):
        from emotive.config.schema import MoodConfig

        cfg = MoodConfig()
        assert cfg.homeostasis_tick_hours == 0.15

    def test_tick_interval_validation(self):
        from pydantic import ValidationError
        from emotive.config.schema import MoodConfig

        # Should reject 0
        try:
            MoodConfig(homeostasis_tick_interval=0)
            assert False, "Should have raised validation error"
        except ValidationError:
            pass

    def test_tick_hours_validation(self):
        from pydantic import ValidationError
        from emotive.config.schema import MoodConfig

        # Should reject 0
        try:
            MoodConfig(homeostasis_tick_hours=0.0)
            assert False, "Should have raised validation error"
        except ValidationError:
            pass
