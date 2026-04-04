"""Locus Coeruleus — norepinephrine arousal modulation.

Controls retrieval scope and encoding threshold based on arousal level.
High arousal (surprise/stress) → narrow, intense retrieval.
Low arousal (relaxed) → broad, associative retrieval.

Already partially implemented in hippocampus/encoding.py (dynamic threshold).
This module extends to retrieval scope modulation.

Brain analog: locus coeruleus, norepinephrine release.
Sources: Frontiers fpsyt.2020.601519, eLife 52059.
"""

from __future__ import annotations

from emotive.logging import get_logger

logger = get_logger("locus_coeruleus")


def compute_retrieval_scope(
    prediction_error: float,
    emotional_intensity: float = 0.0,
    energy: float = 1.0,
) -> dict:
    """Compute retrieval scope parameters from arousal state.

    Returns dict with scope modifiers:
      - candidate_multiplier: how many candidates to fetch (1.0 = normal)
      - threshold_modifier: retrieval threshold adjustment
      - scope: "narrow" | "focused" | "broad"

    Low arousal (< 0.3): BROAD — many candidates, weak threshold
      → mind-wandering, loose associations, DMN-like
    Medium (0.3-0.6): FOCUSED — standard
      → task-relevant, balanced
    High (> 0.6): NARROW — few candidates, strong threshold
      → only most salient/emotional memories
    """
    # Combined arousal from prediction error + emotion
    arousal = prediction_error * 0.6 + emotional_intensity * 0.4

    # Low energy amplifies the effect of arousal
    if energy < 0.3:
        arousal *= 1.3  # tired = more reactive

    if arousal < 0.3:
        return {
            "scope": "broad",
            "candidate_multiplier": 1.5,
            "threshold_modifier": -0.1,  # lower threshold = more comes through
        }
    elif arousal > 0.6:
        return {
            "scope": "narrow",
            "candidate_multiplier": 0.6,
            "threshold_modifier": 0.15,  # higher threshold = only strong signals
        }
    else:
        return {
            "scope": "focused",
            "candidate_multiplier": 1.0,
            "threshold_modifier": 0.0,
        }
