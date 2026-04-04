"""System 2 gate: decides when to engage expanded inner speech.

Most exchanges don't need deliberation -- warmth bypass lets them
through without an inner LLM call. Only high-conflict, high-surprise,
or emotionally confusing situations trigger System 2.

Brain analog: ACC conflict monitoring -> dlPFC deliberation trigger.
"""

from __future__ import annotations

from emotive.config.schema import InnerSpeechConfig


def should_engage_system2(
    social_bonding: float,
    intensity: float,
    prediction_error: float,
    conflict_score: float,
    emotional_clarity: float,
    user_state: str | None,
    config: InnerSpeechConfig,
) -> tuple[bool, str]:
    """Decide whether to engage expanded inner speech (System 2).

    Returns:
        (should_engage, reason) where reason explains the decision.

    Warmth bypass: high social bonding + low intensity + no conflict -> skip.
    Triggers: high prediction error, high conflict, low emotional clarity,
    confrontational user.
    """
    # Warmth bypass: safe, warm, low-stakes -> skip System 2
    if (
        social_bonding > config.warmth_bypass_threshold
        and intensity < config.system2_intensity_threshold
        and conflict_score < config.system2_conflict_threshold
        and user_state not in ("confrontational", "testing")
    ):
        return False, "warmth_bypass"

    # Trigger conditions (any one is sufficient)
    if conflict_score > config.system2_conflict_threshold:
        return True, "conflict"

    if prediction_error > config.system2_prediction_error_threshold:
        return True, "prediction_error"

    if intensity > 0.7:
        return True, "high_intensity"

    if emotional_clarity < 0.4:
        return True, "low_clarity"

    if user_state in ("confrontational", "testing"):
        return True, "user_state"

    return False, "no_trigger"
