"""Self-Output Appraisal subsystem: post-response monitoring.

Evaluates the generated response for tone alignment with the inner
voice nudge and detects novel concepts (discovery). Publishes events
for misalignment and discovery.

Brain analog: ACC + OFC post-action monitoring -- "was that right?"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    DISCOVERY_DETECTED,
    SELF_APPRAISAL_COMPLETE,
    TONE_MISALIGNMENT,
)
from emotive.subsystems import Subsystem

from .tone_monitor import check_tone_alignment
from emotive.subsystems.dmn.discovery import detect_discovery

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("acc.self_appraisal")


class SelfAppraisal(Subsystem):
    """Post-response self-appraisal: tone alignment + discovery detection."""

    name = "self_appraisal"

    def evaluate(
        self,
        response_text: str,
        response_embedding: list[float],
        nudge: str,
        inner_speech: str | None,
        recalled_embeddings: list[list[float]],
    ) -> dict:
        """Evaluate the generated response.

        Args:
            response_text: The generated response text.
            response_embedding: Embedding of the response.
            nudge: The inner voice nudge that guided generation.
            inner_speech: The inner speech thought (if any).
            recalled_embeddings: Embeddings of recalled memories.

        Returns:
            Dict with tone_alignment (float) and discovery (bool).
        """
        tone = check_tone_alignment(response_text, nudge)
        discovery = detect_discovery(response_embedding, recalled_embeddings)

        if tone < 0.3:
            self._bus.publish(
                TONE_MISALIGNMENT,
                {"nudge": nudge, "alignment": tone},
            )
            logger.info("Tone misalignment: nudge=%s alignment=%.2f", nudge, tone)

        if discovery:
            self._bus.publish(
                DISCOVERY_DETECTED,
                {"response_preview": response_text[:100]},
            )
            logger.info("Discovery detected in response")

        result = {"tone_alignment": tone, "discovery": discovery}

        self._bus.publish(SELF_APPRAISAL_COMPLETE, result)

        return result
