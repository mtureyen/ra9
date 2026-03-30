"""Appraisal engine — evaluates events and produces emotional responses.

The appraisal engine is the heart of the emotional system. It takes an event
description and produces an appraisal vector (5 dimensions), maps it to
primary/secondary emotions, calculates intensity, and detects formative events.

In Phase 1, the LLM provides the appraisal vector (self-assessment).
A rule-based fallback exists for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AppraisalVector:
    """5-dimensional appraisal of an event."""

    goal_relevance: float  # 0-1: how much does this matter?
    novelty: float  # 0-1: how unexpected is this?
    valence: float  # 0-1: positive (1) or negative (0)?
    agency: float  # 0-1: self-caused (1) or external (0)?
    social_significance: float  # 0-1: how relational is this?

    def validate(self) -> None:
        for name in [
            "goal_relevance", "novelty", "valence", "agency", "social_significance",
        ]:
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be 0-1, got {val}")


@dataclass
class AppraisalResult:
    """Complete result of appraising an event."""

    vector: AppraisalVector
    primary_emotion: str
    secondary_emotions: list[str]
    intensity: float  # 0-1
    half_life_minutes: float
    is_formative: bool
    decay_rate: float  # per-minute


# Emotion scoring patterns: each emotion is scored against the appraisal vector.
# Highest score wins as primary, next 1-2 as secondary.
EMOTION_PATTERNS: dict[str, dict[str, tuple[str, float]]] = {
    "joy": {
        "requires": ("valence", 0.5),  # valence must be > 0.5
        "weights": {
            "valence": 0.4,
            "goal_relevance": 0.4,
            "novelty": 0.2,
        },
    },
    "sadness": {
        "requires": ("valence", -0.5),  # valence must be < 0.5
        "weights": {
            "valence": -0.4,  # inverted: low valence = high sadness
            "goal_relevance": 0.4,
            "agency": -0.2,  # low agency amplifies sadness
        },
    },
    "anger": {
        "requires": ("valence", -0.5),
        "weights": {
            "valence": -0.3,
            "goal_relevance": 0.3,
            "agency": -0.2,  # external cause
            "social_significance": 0.2,
        },
    },
    "fear": {
        "requires": ("valence", -0.5),
        "weights": {
            "valence": -0.3,
            "novelty": 0.3,
            "agency": -0.2,  # can't control it
            "goal_relevance": 0.2,
        },
    },
    "surprise": {
        "requires": None,
        "weights": {
            "novelty": 0.6,
            "goal_relevance": 0.2,
            "valence": 0.2,
        },
    },
    "awe": {
        "requires": ("valence", 0.5),
        "weights": {
            "novelty": 0.35,
            "goal_relevance": 0.25,
            "social_significance": 0.2,
            "valence": 0.2,
        },
    },
    "disgust": {
        "requires": ("valence", -0.5),
        "weights": {
            "valence": -0.4,
            "social_significance": 0.3,
            "agency": -0.3,
        },
    },
    "trust": {
        "requires": ("valence", 0.5),
        "weights": {
            "social_significance": 0.4,
            "valence": 0.3,
            "goal_relevance": 0.2,
            "novelty": -0.1,  # familiarity helps trust
        },
    },
}


def calculate_intensity(
    vector: AppraisalVector,
    sensitivity: float = 0.5,
) -> float:
    """Calculate episode intensity from appraisal vector.

    Formula from architecture doc:
    base = (goal_relevance * 0.3) + (novelty * 0.2) +
           (|valence - 0.5| * 2 * 0.2) + (agency * 0.15) +
           (social_significance * 0.15)
    modulated = base * sensitivity
    """
    base = (
        vector.goal_relevance * 0.3
        + vector.novelty * 0.2
        + abs(vector.valence - 0.5) * 2 * 0.2
        + vector.agency * 0.15
        + vector.social_significance * 0.15
    )
    return min(max(base * (0.5 + sensitivity), 0.0), 1.0)


def map_emotions(vector: AppraisalVector) -> tuple[str, list[str]]:
    """Map appraisal vector to primary and secondary emotions."""
    scores: dict[str, float] = {}

    for emotion, pattern in EMOTION_PATTERNS.items():
        req = pattern["requires"]
        if req is not None:
            dim_name, threshold = req
            val = getattr(vector, dim_name)
            if threshold > 0 and val < threshold:
                continue
            if threshold < 0 and val > abs(threshold):
                continue

        score = 0.0
        for dim, weight in pattern["weights"].items():
            val = getattr(vector, dim)
            if weight < 0:
                score += abs(weight) * (1.0 - val)
            else:
                score += weight * val
        scores[emotion] = score

    if not scores:
        return "surprise", []

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary = ranked[0][0]
    secondary = [e for e, s in ranked[1:3] if s > 0.3]

    return primary, secondary


def calculate_half_life(
    intensity: float,
    resilience: float = 0.5,
    base_half_life: float = 30.0,
) -> float:
    """Calculate episode half-life in minutes.

    High resilience = shorter half-life (quicker recovery).
    High intensity = slightly longer half-life (strong emotions linger).
    """
    resilience_factor = 0.5 + resilience  # 0.5 -> 0.5x, 1.0 -> 1.5x speed
    intensity_factor = 1.0 + intensity * 0.5  # high intensity lingers longer
    return base_half_life * intensity_factor / resilience_factor


def run_appraisal(
    vector: AppraisalVector,
    *,
    sensitivity: float = 0.5,
    resilience: float = 0.5,
    base_half_life: float = 30.0,
    formative_threshold: float = 0.8,
) -> AppraisalResult:
    """Full appraisal pipeline: vector -> emotion + intensity + metadata."""
    vector.validate()

    intensity = calculate_intensity(vector, sensitivity)
    primary, secondary = map_emotions(vector)
    half_life = calculate_half_life(intensity, resilience, base_half_life)
    is_formative = intensity > formative_threshold

    # Decay rate derived from half-life: intensity * 0.5^(t/half_life)
    # decay_rate = ln(2) / half_life (per minute)
    import math

    decay_rate = math.log(2) / half_life if half_life > 0 else 1.0

    return AppraisalResult(
        vector=vector,
        primary_emotion=primary,
        secondary_emotions=secondary,
        intensity=intensity,
        half_life_minutes=half_life,
        is_formative=is_formative,
        decay_rate=decay_rate,
    )


def rule_based_appraisal(event: str, source: str) -> AppraisalVector:
    """Naive rule-based fallback when LLM doesn't provide appraisal.

    Uses keyword matching for a rough estimate. Not the primary path —
    Ryo should self-appraise via the experience_event tool.
    """
    text = event.lower()

    # Goal relevance
    goal_keywords = ["important", "matters", "goal", "need", "must", "critical"]
    goal_relevance = 0.3 + 0.4 * any(k in text for k in goal_keywords)

    # Novelty
    novelty_keywords = ["new", "first", "never", "unexpected", "surprise", "discover"]
    novelty = 0.3 + 0.4 * any(k in text for k in novelty_keywords)

    # Valence
    pos_keywords = ["good", "great", "happy", "love", "thank", "beautiful", "trust"]
    neg_keywords = ["bad", "sad", "angry", "hurt", "fear", "wrong", "fail", "lost"]
    pos = sum(1 for k in pos_keywords if k in text)
    neg = sum(1 for k in neg_keywords if k in text)
    valence = 0.5 + (pos - neg) * 0.15
    valence = min(max(valence, 0.0), 1.0)

    # Agency
    agency = 0.5
    if source == "internal_realization":
        agency = 0.8
    elif source == "user_message":
        agency = 0.2

    # Social significance
    social_keywords = ["friend", "trust", "share", "together", "connect", "love"]
    social = 0.3 + 0.4 * any(k in text for k in social_keywords)

    return AppraisalVector(
        goal_relevance=goal_relevance,
        novelty=novelty,
        valence=valence,
        agency=agency,
        social_significance=social,
    )
