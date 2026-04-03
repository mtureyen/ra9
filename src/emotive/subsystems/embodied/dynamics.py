"""Pure functions for embodied state dynamics.

All functions are stateless and independently testable.
All values clamped to [0.0, 1.0].
"""

from __future__ import annotations


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def deplete_energy(current: float, base_rate: float) -> float:
    """Nonlinear energy depletion.

    Above 0.5: cost = base_rate (normal).
    Below 0.5: cost = base_rate * 2 (fatigue accelerates).
    """
    cost = base_rate if current >= 0.5 else base_rate * 2
    return _clamp(current - cost)


def boost_energy(
    current: float, emotion: str, intensity: float, boost_amount: float
) -> float:
    """Joy/awe at intensity > 0.6 gives a small energy boost."""
    if emotion in ("joy", "awe") and intensity > 0.6:
        return _clamp(current + boost_amount)
    return current


def update_cognitive_load(
    current: float, num_recalled: int, prediction_error: float
) -> float:
    """Cognitive load increases with recalled memories and prediction error.

    Each recalled memory adds a small increment; high prediction error
    adds a larger bump. Load decays slightly each call (attention resets).
    """
    # Slight natural decay toward 0
    decayed = current * 0.9

    # Recalled memories add load
    recall_bump = num_recalled * 0.02

    # Prediction error adds load
    error_bump = prediction_error * 0.1

    return _clamp(decayed + recall_bump + error_bump)


def update_comfort(
    current: float, emotion: str, intensity: float, decay_rate: float
) -> float:
    """Comfort tracks emotion dynamics.

    Trust/joy increase comfort. Anger/disgust/fear decrease comfort.
    Otherwise decays slowly toward 0.5 (neutral).
    """
    positive_emotions = {"trust", "joy", "awe"}
    negative_emotions = {"anger", "disgust", "fear", "sadness"}

    if emotion in positive_emotions:
        delta = intensity * 0.05
        return _clamp(current + delta)
    elif emotion in negative_emotions:
        delta = intensity * 0.05
        return _clamp(current - delta)
    else:
        # Decay toward 0.5
        if current > 0.5:
            return _clamp(current - decay_rate)
        elif current < 0.5:
            return _clamp(current + decay_rate)
        return current


def recover_energy(current: float, hours_elapsed: float) -> float:
    """Between-session energy recovery.

    Energy recovers toward 1.0 over time when the system is idle.
    Recovery rate: ~0.1 per hour, diminishing as energy approaches 1.0.
    """
    if hours_elapsed <= 0:
        return current

    recovery_rate = 0.1
    deficit = 1.0 - current
    recovery = deficit * (1 - (1 - recovery_rate) ** hours_elapsed)
    return _clamp(current + recovery)
