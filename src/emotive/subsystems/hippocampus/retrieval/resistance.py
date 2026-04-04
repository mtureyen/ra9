"""Memory resistance — E20.

Identity-threatening memories have elevated retrieval thresholds
that lower with trust + safety. Difficult memories approach on
your terms, not forced into consciousness.

Brain analog: ACC + dlPFC inhibitory control of threatening memories.
Sources: Heidegger, Being and Time (thrownness); van der Kolk (trauma).
"""

from __future__ import annotations

import uuid

from emotive.logging import get_logger

logger = get_logger("hippocampus.retrieval.resistance")


def compute_resistance_threshold(
    identity_resistance: float,
    person_trust: float = 0.5,
    energy: float = 0.5,
    comfort: float = 0.5,
    is_direct_inquiry: bool = False,
) -> float:
    """Compute the effective retrieval threshold multiplier for a resistant memory.

    Returns a multiplier > 1.0 (harder to retrieve) or < 1.0 (lowered resistance).

    Resistance is lowered by:
    - High trust (> 0.7): safety enables approach
    - High energy (> 0.5): resources for difficult processing
    - High comfort (> 0.6): emotional safety
    - Direct inquiry: "tell me about X" lowers resistance

    Resistance decreases by 15% per safe exposure (habituation).
    """
    if identity_resistance <= 0:
        return 1.0  # no resistance

    # Base: resistance raises threshold by 50%
    multiplier = 1.0 + identity_resistance * 0.5

    # Safety conditions lower resistance
    if person_trust > 0.7 and energy > 0.5 and comfort > 0.6:
        multiplier *= 0.8  # safe context

    # Direct inquiry lowers resistance further
    if is_direct_inquiry:
        multiplier *= 0.6

    return max(multiplier, 0.5)  # never fully removes resistance


def update_resistance_after_safe_exposure(
    memory_metadata: dict,
    person_trust: float,
    comfort: float,
) -> dict:
    """Reduce identity_resistance after successful safe retrieval.

    Called after a resistant memory surfaces in safe conditions.
    Each safe exposure reduces resistance by 15% (habituation).
    """
    resistance = memory_metadata.get("identity_resistance", 0)
    if resistance <= 0:
        return memory_metadata

    if person_trust > 0.6 and comfort > 0.5:
        new_resistance = resistance * 0.85  # 15% reduction
        if new_resistance < 0.1:
            new_resistance = 0  # fully habituated
        memory_metadata["identity_resistance"] = new_resistance
        logger.info(
            "Memory resistance reduced: %.2f → %.2f (safe exposure)",
            resistance, new_resistance,
        )

    return memory_metadata
