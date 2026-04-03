"""Inner Voice subsystem: condensed felt nudge.

Maps current cognitive/emotional/relational state to a single-word
felt nudge that guides response generation. Fast, rule-based, no LLM.

Brain analog: anterior insula felt sense -- the wordless "feel" that
orients behavior before deliberation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import INNER_VOICE_NUDGE
from emotive.subsystems import Subsystem

from .rules import compute_nudge

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus
    from emotive.subsystems.metacognition.markers import MetacognitiveMarkers

logger = get_logger("inner_voice")


class InnerVoice(Subsystem):
    """Condensed inner voice -- single-word felt nudge."""

    name = "inner_voice"

    def nudge(
        self,
        mood: dict,
        trust_level: str,
        user_state: str | None,
        metacog: "MetacognitiveMarkers",
        energy: float,
        comfort: float,
    ) -> str:
        """Compute and publish the felt nudge.

        Args:
            mood: Current mood dimensions dict.
            trust_level: "unknown", "known", "trusted", "core".
            user_state: From social perception (or None).
            metacog: MetacognitiveMarkers from metacognition subsystem.
            energy: Embodied energy level (0-1).
            comfort: Embodied comfort level (0-1).

        Returns:
            Single-word nudge string.
        """
        result = compute_nudge(mood, trust_level, user_state, metacog, energy, comfort)
        self._bus.publish(INNER_VOICE_NUDGE, {"nudge": result})
        logger.info("Inner voice nudge: %s", result)
        return result
