"""Metacognition subsystem: awareness of own cognitive state.

Evaluates how confident the system is in its memories, how clear its
emotions are, and how familiar the current topic is. Feeds markers
into inner voice and system prompt.

Brain analog: anterior insula + dACC metacognitive monitoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emotive.logging import get_logger
from emotive.runtime.event_bus import METACOGNITION_COMPLETE
from emotive.subsystems import Subsystem

from .markers import MetacognitiveMarkers

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus
    from emotive.subsystems.workspace.signals import WorkspaceOutput

logger = get_logger("metacognition")


class Metacognition(Subsystem):
    """Metacognitive monitoring -- awareness of own cognitive state."""

    name = "metacognition"

    def evaluate(
        self,
        recalled_memories: list[dict],
        appraisal: Any,
        workspace_output: "WorkspaceOutput",
    ) -> MetacognitiveMarkers:
        """Evaluate metacognitive markers from current cognitive state.

        Args:
            recalled_memories: Memories returned by retrieval.
            appraisal: AppraisalResult from amygdala.
            workspace_output: Output from global workspace.

        Returns:
            MetacognitiveMarkers with confidence and clarity signals.
        """
        # Memory confidence: average similarity of top recalled memories
        memory_confidence = _compute_memory_confidence(recalled_memories)

        # Emotional clarity: gap between primary and secondary emotion intensity
        emotional_clarity = _compute_emotional_clarity(appraisal)

        # Knowledge confidence: based on recall count and average similarity
        knowledge_confidence = _compute_knowledge_confidence(recalled_memories)

        markers = MetacognitiveMarkers(
            memory_confidence=memory_confidence,
            emotional_clarity=emotional_clarity,
            knowledge_confidence=knowledge_confidence,
        )

        self._bus.publish(
            METACOGNITION_COMPLETE,
            {
                "memory_confidence": markers.memory_confidence,
                "emotional_clarity": markers.emotional_clarity,
                "knowledge_confidence": markers.knowledge_confidence,
                "felt": markers.to_felt_description(),
            },
        )

        logger.info(
            "Metacognition: mem=%.2f clarity=%.2f know=%.2f | %s",
            markers.memory_confidence,
            markers.emotional_clarity,
            markers.knowledge_confidence,
            markers.to_felt_description() or "(clear)",
        )

        return markers


def _compute_memory_confidence(recalled_memories: list[dict]) -> float:
    """Average retrieval_score of recalled memories. 0.0 if none."""
    if not recalled_memories:
        return 0.0
    scores = [
        float(m.get("retrieval_score", m.get("similarity", 0.0)))
        for m in recalled_memories
    ]
    return sum(scores) / len(scores) if scores else 0.0


def _compute_emotional_clarity(appraisal: Any) -> float:
    """Clarity = gap between primary intensity and secondary emotion intensity.

    High clarity: dominant emotion is clearly stronger than alternatives.
    Low clarity: mixed feelings, multiple emotions competing.
    """
    intensity = getattr(appraisal, "intensity", 0.0)
    secondary = getattr(appraisal, "secondary_emotions", [])

    if not secondary:
        # No secondary emotions -> very clear
        return min(1.0, intensity + 0.3)

    # Approximate: more secondary emotions = less clarity
    # Clarity decreases with number of competing emotions
    penalty = len(secondary) * 0.2
    return max(0.0, min(1.0, intensity - penalty))


def _compute_knowledge_confidence(recalled_memories: list[dict]) -> float:
    """Confidence based on number and quality of recalled memories.

    Many high-quality recalls = familiar territory.
    Few or low-quality recalls = unfamiliar.
    """
    if not recalled_memories:
        return 0.0

    count_factor = min(len(recalled_memories) / 5.0, 1.0)  # 5+ memories = full confidence
    scores = [
        float(m.get("retrieval_score", m.get("similarity", 0.0)))
        for m in recalled_memories
    ]
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return min(1.0, (count_factor + avg_score) / 2.0)
