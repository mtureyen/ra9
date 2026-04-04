"""Metacognitive markers: confidence and clarity signals.

These markers represent the system's awareness of its own cognitive
state -- how confident it is in its memories, how clear its emotions
are, how familiar the territory is.

Brain analog: anterior insula + dACC metacognitive monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetacognitiveMarkers:
    """Metacognitive state: confidence and clarity signals."""

    memory_confidence: float  # 0-1, from avg similarity of recalled memories
    emotional_clarity: float  # 0-1, from gap between primary and secondary emotion
    knowledge_confidence: float  # 0-1, from recall count and relevance

    def to_felt_description(self) -> str:
        """Convert markers to felt description for system prompt.

        Returns a natural-language description of metacognitive uncertainty.
        Empty string if everything is clear and confident.
        """
        parts: list[str] = []
        if self.memory_confidence < 0.4:
            parts.append("I'm not sure I remember this well")
        if self.emotional_clarity < 0.4:
            parts.append("My feelings about this are mixed")
        if self.knowledge_confidence < 0.3:
            parts.append("This is unfamiliar territory")
        return ". ".join(parts) if parts else ""
