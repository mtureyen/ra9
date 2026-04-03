"""Default Mode Network subsystem: self-schema generation + enhanced DMN.

Regenerates Ryo's self-concept from memory patterns. The self-schema
is a weighted data structure stored in RAM and injected into every
LLM context. It's regenerated after consolidation, not retrieved.

Enhanced DMN adds spontaneous mid-session thoughts and end-session
reflection via thin LLM calls.

Brain analog: DMN (mPFC, PCC, precuneus, TPJ, hippocampus) —
continuously re-computing self-referential processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    CONSOLIDATION_COMPLETED,
    DMN_FLASH,
    SELF_SCHEMA_REGENERATED,
)
from emotive.subsystems import Subsystem

from .reflection import build_reflection_prompt, build_spontaneous_thought_prompt
from .schema import SelfSchema, regenerate_schema
from .spontaneous import find_cross_memory_connection, should_flash

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.config.schema import DMNEnhancedConfig
    from emotive.runtime.event_bus import EventBus

logger = get_logger("dmn")


class DefaultModeNetwork(Subsystem):
    """Self-schema generation subsystem — the DMN analog."""

    name = "dmn"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        self._schema: SelfSchema | None = None

    def _register_handlers(self) -> None:
        """Auto-regenerate self-schema after consolidation."""
        self._bus.subscribe(CONSOLIDATION_COMPLETED, self._on_consolidation)

    def _on_consolidation(self, event_type: str, data: dict) -> None:
        """Event handler: regenerate after consolidation completes."""
        try:
            self.regenerate()
        except Exception:
            logger.exception("Failed to regenerate self-schema after consolidation")

    def regenerate(self) -> SelfSchema:
        """Full regeneration from current memory state."""
        config = self._app.config_manager.get()
        session = self._app.session_factory()
        try:
            self._schema = regenerate_schema(
                session,
                max_traits=config.self_schema.max_traits,
                max_core_facts=config.self_schema.max_core_facts,
                max_values=config.self_schema.max_values,
            )
            self._bus.publish(
                SELF_SCHEMA_REGENERATED,
                {
                    "traits_count": len(self._schema.traits),
                    "core_facts_count": len(self._schema.core_facts),
                    "values_count": len(self._schema.active_values),
                    "persons_count": len(self._schema.person_context),
                },
            )
            logger.info(
                "Self-schema regenerated: %d traits, %d facts, %d values",
                len(self._schema.traits),
                len(self._schema.core_facts),
                len(self._schema.active_values),
            )
            return self._schema
        finally:
            session.close()

    def spontaneous_flash(
        self,
        memories: list[dict],
        embodied_energy: float,
        config: "DMNEnhancedConfig",
    ) -> str | None:
        """Attempt a mid-session spontaneous thought.

        Checks probability gate and energy, then tries to find a
        cross-memory connection worth noting.

        Returns:
            A description of the connection, or None if no flash.
        """
        if not should_flash(config.flash_probability, embodied_energy):
            return None

        pair = find_cross_memory_connection(memories, self._app.embedding_service)
        if pair is None:
            return None

        mem_a, mem_b = pair
        content_a = mem_a.get("content", "?")[:80]
        content_b = mem_b.get("content", "?")[:80]
        thought = f"Something connects '{content_a}' and '{content_b}'..."

        self._bus.publish(DMN_FLASH, {"thought": thought})
        logger.info("DMN flash: %s", thought)
        return thought

    async def end_session_reflection(
        self,
        conversation_summary: str,
        mood: dict,
        llm: Any,
    ) -> str | None:
        """Generate a reflection at session end via thin LLM call.

        Args:
            conversation_summary: Summary of the session.
            mood: Current mood dimensions.
            llm: LLM adapter with async generate method.

        Returns:
            Reflection string, or None on failure.
        """
        schema_summary = ""
        if self._schema:
            traits = ", ".join(
                f"{t}: {w:.1f}" for t, w in list(self._schema.traits.items())[:5]
            )
            schema_summary = f"Traits: {traits}"

        system, messages = build_reflection_prompt(
            conversation_summary, mood, schema_summary
        )

        try:
            reflection = await llm.generate(system, messages)
            reflection = reflection.strip()[:300]
            logger.info("Session reflection: %s", reflection)
            return reflection
        except Exception:
            logger.exception("End-session reflection LLM call failed")
            return None

    @property
    def current(self) -> SelfSchema | None:
        """Current in-RAM self-schema. None if not yet generated."""
        return self._schema
