"""Tests for the Embodied State subsystem (Phase 2.5)."""

import pytest

from emotive.subsystems.embodied.dynamics import (
    boost_energy,
    deplete_energy,
    recover_energy,
    update_cognitive_load,
    update_comfort,
)


class TestEnergyDepletion:
    def test_nonlinear_above_half(self):
        """Above 0.5: cost = base_rate."""
        result = deplete_energy(0.8, 0.02)
        assert result == pytest.approx(0.78)

    def test_nonlinear_below_half(self):
        """Below 0.5: cost = base_rate * 2 (doubles)."""
        result = deplete_energy(0.4, 0.02)
        assert result == pytest.approx(0.36)

    def test_depletion_faster_below_half(self):
        """Energy depletes faster below 0.5 than above."""
        above = 0.6 - deplete_energy(0.6, 0.02)
        below = 0.4 - deplete_energy(0.4, 0.02)
        assert below > above

    def test_energy_never_negative(self):
        result = deplete_energy(0.01, 0.5)
        assert result == 0.0

    def test_energy_never_above_one(self):
        result = boost_energy(0.99, "joy", 0.9, 0.5)
        assert result == 1.0


class TestEnergyBoost:
    def test_joy_high_intensity_boosts(self):
        result = boost_energy(0.5, "joy", 0.8, 0.03)
        assert result == pytest.approx(0.53)

    def test_awe_high_intensity_boosts(self):
        result = boost_energy(0.5, "awe", 0.7, 0.03)
        assert result == pytest.approx(0.53)

    def test_joy_low_intensity_no_boost(self):
        """Joy at intensity <= 0.6 does NOT boost."""
        result = boost_energy(0.5, "joy", 0.5, 0.03)
        assert result == 0.5

    def test_sadness_no_boost(self):
        result = boost_energy(0.5, "sadness", 0.9, 0.03)
        assert result == 0.5

    def test_anger_no_boost(self):
        result = boost_energy(0.5, "anger", 0.9, 0.03)
        assert result == 0.5


class TestCognitiveLoad:
    def test_increases_with_recalled_memories(self):
        result = update_cognitive_load(0.0, num_recalled=5, prediction_error=0.0)
        assert result > 0.0

    def test_increases_with_prediction_error(self):
        result = update_cognitive_load(0.0, num_recalled=0, prediction_error=0.8)
        assert result > 0.0

    def test_natural_decay(self):
        """Load decays even with no new input."""
        result = update_cognitive_load(0.5, num_recalled=0, prediction_error=0.0)
        assert result < 0.5

    def test_clamped_to_one(self):
        result = update_cognitive_load(0.95, num_recalled=10, prediction_error=1.0)
        assert result <= 1.0


class TestComfort:
    def test_trust_increases_comfort(self):
        result = update_comfort(0.5, "trust", 0.7, 0.01)
        assert result > 0.5

    def test_joy_increases_comfort(self):
        result = update_comfort(0.5, "joy", 0.7, 0.01)
        assert result > 0.5

    def test_anger_decreases_comfort(self):
        result = update_comfort(0.5, "anger", 0.7, 0.01)
        assert result < 0.5

    def test_neutral_decays_toward_half(self):
        high = update_comfort(0.8, "neutral", 0.5, 0.01)
        assert high < 0.8
        low = update_comfort(0.2, "neutral", 0.5, 0.01)
        assert low > 0.2

    def test_clamped_to_bounds(self):
        assert update_comfort(0.01, "anger", 1.0, 0.01) >= 0.0
        assert update_comfort(0.99, "trust", 1.0, 0.01) <= 1.0


class TestRecovery:
    def test_recovers_toward_one(self):
        result = recover_energy(0.3, hours_elapsed=5.0)
        assert result > 0.3

    def test_full_energy_stays_full(self):
        result = recover_energy(1.0, hours_elapsed=10.0)
        assert result == 1.0

    def test_zero_hours_no_recovery(self):
        result = recover_energy(0.3, hours_elapsed=0.0)
        assert result == 0.3

    def test_long_idle_near_full(self):
        result = recover_energy(0.1, hours_elapsed=100.0)
        assert result > 0.95


class TestDefaultState:
    def test_defaults(self):
        """Default state: energy=1.0, load=0.0, comfort=0.5."""
        from emotive.subsystems.embodied.dynamics import (
            deplete_energy,
            update_cognitive_load,
            update_comfort,
        )

        # Starting from defaults, a neutral exchange barely changes things
        energy = deplete_energy(1.0, 0.02)
        assert energy == pytest.approx(0.98)

        load = update_cognitive_load(0.0, num_recalled=0, prediction_error=0.0)
        assert load == 0.0

        comfort = update_comfort(0.5, "neutral", 0.0, 0.01)
        assert comfort == 0.5
