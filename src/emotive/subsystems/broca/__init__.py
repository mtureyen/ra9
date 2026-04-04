"""Inner Speech subsystem: expanded deliberation via System 2 gate.

When the System 2 gate opens (high conflict, surprise, or emotional
confusion), inner speech generates a brief deliberative thought via
a thin LLM call. When the gate stays closed, no LLM call is made.

Brain analog: dlPFC deliberative processing triggered by ACC conflict.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    INNER_SPEECH_GENERATED,
    SYSTEM2_GATE_BYPASSED,
    SYSTEM2_GATE_OPENED,
)
from emotive.subsystems import Subsystem

from .gate import should_engage_system2
from .prompt import build_inner_speech_prompt

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.config.schema import InnerSpeechConfig
    from emotive.runtime.event_bus import EventBus
    from emotive.subsystems.prefrontal.metacognition.markers import MetacognitiveMarkers
    from emotive.subsystems.workspace.signals import WorkspaceOutput

logger = get_logger("broca")


class InnerSpeech(Subsystem):
    """Expanded inner speech -- System 2 deliberation."""

    name = "inner_speech"
    last_trigger_reason: str | None = None

    async def think(
        self,
        nudge: str,
        appraisal: Any,
        user_message: str,
        workspace_output: "WorkspaceOutput",
        metacog: "MetacognitiveMarkers",
        llm: Any,
        social_bonding: float,
        conflict_score: float,
        prediction_error: float,
        trust_level: str,
        config: "InnerSpeechConfig",
        privacy_flags: list[str] | None = None,
    ) -> str | None:
        """Run expanded inner speech if System 2 gate opens.

        Returns:
            Thought string if gate opened, None if bypassed.
        """
        engage, reason = should_engage_system2(
            social_bonding,
            appraisal.intensity,
            prediction_error,
            conflict_score,
            metacog.emotional_clarity,
            getattr(appraisal, "user_state", None),
            config,
        )

        self.last_trigger_reason = reason if engage else None

        if not engage:
            self._bus.publish(SYSTEM2_GATE_BYPASSED, {"reason": reason})
            logger.info("System 2 gate bypassed: %s", reason)
            return None

        self._bus.publish(SYSTEM2_GATE_OPENED, {"reason": reason})
        logger.info("System 2 gate opened: %s", reason)

        system, messages = build_inner_speech_prompt(
            nudge,
            appraisal.primary_emotion,
            appraisal.intensity,
            getattr(appraisal, "user_state", None),
            user_message,
            trust_level,
            privacy_flags=privacy_flags,
        )

        try:
            thought = await llm.generate(system, messages)
            thought = thought.strip()[:200]  # Cap length
            self._bus.publish(
                INNER_SPEECH_GENERATED,
                {"thought": thought, "reason": reason},
            )
            logger.info("Inner speech: %s", thought)
            return thought
        except Exception:
            logger.exception("Inner speech LLM call failed")
            return None
