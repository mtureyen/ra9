"""Tests for the appraisal engine."""


import pytest

from emotive.layers.appraisal import (
    AppraisalVector,
    calculate_half_life,
    calculate_intensity,
    map_emotions,
    rule_based_appraisal,
    run_appraisal,
)

# --- AppraisalVector ---


def test_appraisal_vector_validates_range():
    v = AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5)
    v.validate()  # should not raise


def test_appraisal_vector_rejects_out_of_range():
    v = AppraisalVector(1.5, 0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="goal_relevance"):
        v.validate()


def test_appraisal_vector_rejects_negative():
    v = AppraisalVector(-0.1, 0.5, 0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="goal_relevance"):
        v.validate()


# --- Intensity ---


def test_intensity_high_appraisal():
    v = AppraisalVector(0.9, 0.8, 0.9, 0.7, 0.9)
    intensity = calculate_intensity(v, sensitivity=0.5)
    assert intensity > 0.5


def test_intensity_low_appraisal():
    v = AppraisalVector(0.1, 0.1, 0.5, 0.1, 0.1)
    intensity = calculate_intensity(v, sensitivity=0.5)
    assert intensity < 0.3


def test_intensity_scales_with_sensitivity():
    v = AppraisalVector(0.5, 0.5, 0.5, 0.5, 0.5)
    low_sens = calculate_intensity(v, sensitivity=0.2)
    high_sens = calculate_intensity(v, sensitivity=0.9)
    assert high_sens > low_sens


def test_intensity_clamped_to_one():
    v = AppraisalVector(1.0, 1.0, 1.0, 1.0, 1.0)
    intensity = calculate_intensity(v, sensitivity=1.0)
    assert intensity <= 1.0


# --- Emotion mapping ---


def test_positive_event_maps_to_positive_emotion():
    v = AppraisalVector(0.8, 0.5, 0.9, 0.5, 0.8)
    primary, _ = map_emotions(v)
    assert primary in ("joy", "trust", "awe")


def test_negative_event_maps_to_negative_emotion():
    v = AppraisalVector(0.8, 0.5, 0.1, 0.1, 0.8)
    primary, _ = map_emotions(v)
    assert primary in ("sadness", "anger", "fear", "disgust")


def test_novel_event_maps_to_surprise_or_awe():
    v = AppraisalVector(0.5, 0.95, 0.6, 0.5, 0.5)
    primary, _ = map_emotions(v)
    assert primary in ("surprise", "awe", "joy")


def test_social_positive_maps_to_trust():
    v = AppraisalVector(0.7, 0.3, 0.8, 0.5, 0.95)
    primary, _ = map_emotions(v)
    assert primary == "trust"


def test_secondary_emotions_returned():
    v = AppraisalVector(0.8, 0.6, 0.8, 0.3, 0.9)
    _, secondary = map_emotions(v)
    assert isinstance(secondary, list)


# --- Half-life ---


def test_high_resilience_shorter_half_life():
    h_low = calculate_half_life(0.5, resilience=0.2)
    h_high = calculate_half_life(0.5, resilience=0.9)
    assert h_high < h_low


def test_high_intensity_longer_half_life():
    h_low = calculate_half_life(0.2, resilience=0.5)
    h_high = calculate_half_life(0.9, resilience=0.5)
    assert h_high > h_low


# --- Full appraisal ---


def test_run_appraisal_produces_complete_result():
    v = AppraisalVector(0.7, 0.5, 0.8, 0.3, 0.6)
    result = run_appraisal(v, sensitivity=0.5, resilience=0.5)
    assert result.primary_emotion
    assert 0 < result.intensity <= 1
    assert result.half_life_minutes > 0
    assert result.decay_rate > 0
    assert isinstance(result.is_formative, bool)


def test_run_appraisal_formative_detection():
    # High everything should produce formative
    v = AppraisalVector(0.95, 0.9, 0.95, 0.8, 0.95)
    result = run_appraisal(v, sensitivity=0.8, formative_threshold=0.8)
    assert result.is_formative is True


def test_run_appraisal_not_formative_for_low_intensity():
    v = AppraisalVector(0.2, 0.2, 0.5, 0.2, 0.2)
    result = run_appraisal(v, sensitivity=0.3, formative_threshold=0.8)
    assert result.is_formative is False


# --- Rule-based fallback ---


def test_rule_based_positive_event():
    v = rule_based_appraisal("This is a great and beautiful discovery", "user_message")
    assert v.valence > 0.5


def test_rule_based_negative_event():
    v = rule_based_appraisal("This is bad and I feel sad about the failure", "user_message")
    assert v.valence < 0.5


def test_rule_based_agency_from_source():
    v_ext = rule_based_appraisal("something happened", "user_message")
    v_int = rule_based_appraisal("something happened", "internal_realization")
    assert v_int.agency > v_ext.agency


def test_rule_based_social_keywords():
    v = rule_based_appraisal("I trust my friend and we share everything", "user_message")
    assert v.social_significance > 0.5
