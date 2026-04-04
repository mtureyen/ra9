"""Somatic markers — body state shapes memory retrieval (E14).

The insular cortex (interoception) and vmPFC store associations between
situations and body-state changes. Low comfort → threat memories more
accessible. High energy → action-oriented memories. Low energy →
defenses lower, suppressed memories break through.

Brain analog: vmPFC somatic marker pathway (Damasio).
Sources: Damasio (1996), Philosophical Transactions; ScienceDirect.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SomaticBias:
    """Retrieval biases derived from current body state."""

    # Boost threat-related memories when comfort is low
    threat_boost: float = 0.0
    # Boost action/procedural memories when energy is high
    action_boost: float = 0.0
    # Weaken intentional suppression when energy is low
    suppression_weakening: float = 0.0
    # Lower spontaneous recall threshold when energy is low
    spontaneous_threshold_reduction: float = 0.0
    # Narrow retrieval to strongest signals when load is high
    retrieval_narrowing: bool = False


def compute_somatic_bias(
    energy: float = 1.0,
    cognitive_load: float = 0.0,
    comfort: float = 0.5,
) -> SomaticBias:
    """Compute retrieval biases from current embodied state.

    The body shapes what you remember:
    - Low comfort → hypervigilance, threat memories surface
    - High energy → ready for action, procedural memories boost
    - Low energy → defenses down, suppressed memories break through
    - High cognitive load → only strongest signals get through
    """
    bias = SomaticBias()

    # Low comfort → threat memories more accessible
    if comfort < 0.4:
        bias.threat_boost = (0.4 - comfort) * 0.5  # up to 0.2

    # High energy → action/procedural boost
    if energy > 0.7:
        bias.action_boost = (energy - 0.7) * 0.5  # up to 0.15

    # Low energy → defenses lower
    if energy < 0.3:
        bias.suppression_weakening = (0.3 - energy) * 1.67  # up to 0.5
        bias.spontaneous_threshold_reduction = (0.3 - energy) * 0.33  # up to 0.1

    # High cognitive load → narrow retrieval
    if cognitive_load > 0.7:
        bias.retrieval_narrowing = True

    return bias


def apply_somatic_bias_to_score(
    score: float,
    memory_tags: list[str],
    memory_type: str,
    primary_emotion: str | None,
    bias: SomaticBias,
) -> float:
    """Apply somatic marker biases to a memory's retrieval score."""
    threat_emotions = {"fear", "anger", "disgust", "sadness"}
    action_tags = {"behavioral_coaching", "procedural"}

    # Threat boost: memories with threat-related emotions
    if bias.threat_boost > 0:
        if primary_emotion in threat_emotions:
            score += bias.threat_boost
        if any(t in threat_emotions for t in memory_tags):
            score += bias.threat_boost * 0.5

    # Action boost: procedural memories
    if bias.action_boost > 0:
        if memory_type == "procedural" or any(t in action_tags for t in memory_tags):
            score += bias.action_boost

    return score
