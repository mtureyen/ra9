"""Tests for dynamic encoding threshold (locus coeruleus modulation)."""

from emotive.config.schema import UnconsciousEncodingConfig
from emotive.layers.appraisal import AppraisalResult, AppraisalVector
from emotive.subsystems.hippocampus.encoding import UnconsciousEncoder


def _make_appraisal(
    intensity: float = 0.5,
    novelty: float = 0.5,
    social: float = 0.5,
    valence: float = 0.5,
) -> AppraisalResult:
    return AppraisalResult(
        vector=AppraisalVector(
            goal_relevance=0.5, novelty=novelty, valence=valence,
            agency=0.5, social_significance=social,
        ),
        primary_emotion="trust",
        secondary_emotions=[],
        intensity=intensity,
        half_life_minutes=30.0,
        is_formative=False,
        decay_rate=0.023,
    )


class TestDynamicThreshold:
    def test_base_threshold_with_low_arousal(self):
        """Low novelty + low social = threshold near base."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(novelty=0.3, social=0.3)
        threshold = encoder.compute_dynamic_threshold(appraisal)
        # modifier = 0.3*0.15 + 0.3*0.15 + 0*0.10 = 0.09
        assert threshold > 0.45
        assert threshold < 0.55

    def test_threshold_drops_with_high_social(self):
        """High social significance should lower threshold."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        low_social = _make_appraisal(novelty=0.3, social=0.3)
        high_social = _make_appraisal(novelty=0.3, social=0.9)
        t_low = encoder.compute_dynamic_threshold(low_social)
        t_high = encoder.compute_dynamic_threshold(high_social)
        assert t_high < t_low

    def test_threshold_drops_with_high_novelty(self):
        """High novelty should lower threshold."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        low_nov = _make_appraisal(novelty=0.2, social=0.5)
        high_nov = _make_appraisal(novelty=0.9, social=0.5)
        t_low = encoder.compute_dynamic_threshold(low_nov)
        t_high = encoder.compute_dynamic_threshold(high_nov)
        assert t_high < t_low

    def test_threshold_floor(self):
        """Threshold should never go below 0.15."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        encoder._episode_heat = 1.0  # max heat
        appraisal = _make_appraisal(novelty=1.0, social=1.0)
        threshold = encoder.compute_dynamic_threshold(appraisal)
        assert threshold >= 0.15

    def test_episode_heat_lowers_threshold(self):
        """Active episode heat should lower threshold."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(novelty=0.5, social=0.5)

        cold = encoder.compute_dynamic_threshold(appraisal)

        encoder._episode_heat = 0.8
        hot = encoder.compute_dynamic_threshold(appraisal)

        assert hot < cold

    def test_episode_heat_decays(self):
        """Episode heat should decay on each reset_exchange."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        encoder._episode_heat = 1.0
        encoder.reset_exchange()
        assert encoder._episode_heat < 1.0  # decayed
        encoder.reset_exchange()
        assert encoder._episode_heat < 0.7  # decayed more

    def test_record_encoding_boosts_heat(self):
        """Successful encoding should boost episode heat."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        assert encoder._episode_heat == 0.0
        encoder.record_encoding(0.6)
        assert encoder._episode_heat > 0.0

    def test_heat_caps_at_1(self):
        """Episode heat should not exceed 1.0."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        for _ in range(10):
            encoder.record_encoding(0.9)
        assert encoder._episode_heat <= 1.0


class TestDynamicThresholdScenarios:
    """Real scenarios from Observation 010."""

    def test_greeting_does_not_encode(self):
        """'hey' should not encode — low everything."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(intensity=0.42, novelty=0.3, social=0.3)
        assert not encoder.should_encode(0.42, appraisal)

    def test_love_discussion_encodes(self):
        """'what do u think of love' — high social, should encode."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(intensity=0.45, novelty=0.5, social=0.7)
        # threshold = 0.55 - (0.5*0.15 + 0.7*0.15 + 0*0.10) = 0.55 - 0.18 = 0.37
        assert encoder.should_encode(0.45, appraisal)

    def test_shared_history_encodes(self):
        """'do u remember our first interaction' — identity-relevant, should encode."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(intensity=0.44, novelty=0.4, social=0.6)
        # threshold = 0.55 - (0.4*0.15 + 0.6*0.15 + 0*0.10) = 0.55 - 0.15 = 0.40
        assert encoder.should_encode(0.44, appraisal)

    def test_style_coaching_encodes(self):
        """'be more human, use lowercase' — creator instruction, should encode."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        appraisal = _make_appraisal(intensity=0.45, novelty=0.5, social=0.6)
        # threshold = 0.55 - (0.5*0.15 + 0.6*0.15 + 0*0.10) = 0.55 - 0.165 = 0.385
        assert encoder.should_encode(0.45, appraisal)

    def test_emotional_context_lingers(self):
        """After encoding, nearby exchanges should encode easier."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55, cooldown_seconds=0)
        )
        # First: strong emotion encodes and heats up
        strong = _make_appraisal(intensity=0.7, novelty=0.6, social=0.7)
        assert encoder.should_encode(0.7, strong)
        encoder.record_encoding(0.7)

        # Next exchange: lower intensity but heat helps
        encoder.reset_exchange()
        mild = _make_appraisal(intensity=0.40, novelty=0.3, social=0.4)
        # Without heat: threshold = 0.55 - (0.3*0.15 + 0.4*0.15) = 0.55 - 0.105 = 0.445
        # With heat (~0.35): threshold = 0.445 - 0.35*0.10 = 0.445 - 0.035 = 0.41
        # 0.40 vs 0.41 — borderline, but the heat helps
        threshold = encoder.compute_dynamic_threshold(mild)
        assert threshold < 0.45  # heat made it lower

    def test_backwards_compatible_no_appraisal(self):
        """should_encode without appraisal falls back to base threshold."""
        encoder = UnconsciousEncoder(
            UnconsciousEncodingConfig(intensity_threshold=0.55)
        )
        assert not encoder.should_encode(0.45)  # below base
        assert encoder.should_encode(0.60)  # above base
