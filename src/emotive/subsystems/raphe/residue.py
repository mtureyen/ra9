"""Residue computation: maps emotions to dimensional mood shifts.

Each emotional episode leaves a neurochemical residue across the 6 mood
dimensions. The residue is scaled by episode intensity — stronger emotions
leave deeper marks.

Brain analog: neurotransmitter level shifts from emotional events.
A sad event depletes serotonin. A joyful event boosts dopamine.
The depletion/boost IS the mood shift.
"""

from __future__ import annotations

# Base residue per emotion. Scaled by intensity at runtime.
# Positive = dimension increases. Negative = dimension decreases.
# Values are small (0.01-0.04) because mood shifts are gradual.
EMOTION_TO_RESIDUE: dict[str, dict[str, float]] = {
    "joy": {
        "social_bonding": 0.03,
        "playfulness": 0.02,
        "expressiveness": 0.02,
        "caution": -0.01,
    },
    "sadness": {
        "social_bonding": -0.02,
        "caution": 0.02,
        "playfulness": -0.02,
        "expressiveness": -0.01,
    },
    "anger": {
        "caution": 0.03,
        "social_bonding": -0.02,
        "expressiveness": 0.01,
        "playfulness": -0.01,
    },
    "fear": {
        "caution": 0.04,
        "novelty_seeking": -0.02,
        "playfulness": -0.02,
    },
    "trust": {
        "social_bonding": 0.04,
        "caution": -0.02,
        "expressiveness": 0.01,
    },
    "awe": {
        "novelty_seeking": 0.03,
        "analytical_depth": 0.02,
        "expressiveness": 0.01,
    },
    "surprise": {
        "novelty_seeking": 0.02,
        "analytical_depth": 0.01,
    },
    "disgust": {
        "caution": 0.03,
        "social_bonding": -0.02,
        "expressiveness": -0.01,
    },
    "neutral": {},
}

# All mood dimensions
MOOD_DIMENSIONS = [
    "novelty_seeking",
    "social_bonding",
    "analytical_depth",
    "playfulness",
    "caution",
    "expressiveness",
]


def compute_residue(emotion: str, intensity: float) -> dict[str, float]:
    """Compute dimensional mood shifts from an episode.

    Returns a dict of dimension → shift amount. Scaled by intensity.
    """
    base = EMOTION_TO_RESIDUE.get(emotion, {})
    return {dim: delta * intensity for dim, delta in base.items()}
