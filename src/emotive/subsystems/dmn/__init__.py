"""Default Mode Network subsystem: self-schema generation.

Regenerates Ryo's self-concept from memory patterns. The self-schema
is a weighted data structure stored in RAM and injected into every
LLM context. It's regenerated after consolidation, not retrieved.

Brain analog: DMN (mPFC, PCC, precuneus, TPJ, hippocampus) —
continuously re-computing self-referential processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    CONSOLIDATION_COMPLETED,
    SELF_SCHEMA_REGENERATED,
)
from emotive.subsystems import Subsystem

from .schema import SelfSchema, regenerate_schema

if TYPE_CHECKING:
    from emotive.app_context import AppContext
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

    @property
    def current(self) -> SelfSchema | None:
        """Current in-RAM self-schema. None if not yet generated."""
        return self._schema
