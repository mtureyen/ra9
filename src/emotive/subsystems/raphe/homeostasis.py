"""Homeostasis: mood decays toward temperament baseline.

The brain doesn't "pull" mood back to baseline. Neurotransmitter reuptake
and synthesis naturally return levels to equilibrium. Different
neurotransmitters have different reuptake rates — dopamine (excitement)
clears fast, serotonin (social warmth) lingers.

Brain analog: neurotransmitter reuptake rates. Not gravity — chemistry.
"""

from __future__ import annotations

from .residue import MOOD_DIMENSIONS

# Reuptake rates per dimension (fraction per hour).
# Higher = faster return to baseline.
REUPTAKE_RATES: dict[str, float] = {
    "novelty_seeking": 0.05,    # fast — excitement fades quickly (dopamine)
    "social_bonding": 0.02,     # slow — social warmth lingers (serotonin/oxytocin)
    "analytical_depth": 0.04,   # medium
    "playfulness": 0.05,        # fast — playfulness is fleeting (dopamine)
    "caution": 0.03,            # medium-slow — vigilance persists (norepinephrine)
    "expressiveness": 0.04,     # medium
}


def apply_homeostasis(
    mood: dict[str, float],
    temperament: dict[str, float],
    hours_elapsed: float,
) -> dict[str, float]:
    """Decay each mood dimension toward its temperament baseline.

    Uses exponential decay: each dimension moves toward baseline at its
    own rate. Fast dimensions (novelty, playfulness) snap back quickly.
    Slow dimensions (social_bonding, caution) linger.

    Returns the updated mood dict.
    """
    result = {}
    for dim in MOOD_DIMENSIONS:
        rate = REUPTAKE_RATES.get(dim, 0.03)
        baseline = temperament.get(dim, 0.5)
        current = mood.get(dim, baseline)
        # Exponential decay toward baseline
        result[dim] = current + (baseline - current) * min(rate * hours_elapsed, 1.0)
    return result
