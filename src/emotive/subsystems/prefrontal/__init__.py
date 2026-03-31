"""Prefrontal Cortex subsystem: working memory, conversation buffer, context assembly.

Manages the conversation window, compresses exited turns into gists,
and builds the enriched context the LLM receives.

Brain analog: dorsolateral PFC (working memory) + ventromedial PFC (context integration).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.subsystems import Subsystem

from .buffer import ConversationBuffer, ConversationTurn, compress_to_gist
from .context import build_messages, build_system_prompt

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.layers.appraisal import AppraisalResult
    from emotive.runtime.event_bus import EventBus
    from emotive.subsystems.dmn.schema import SelfSchema

logger = get_logger("prefrontal")


class PrefrontalCortex(Subsystem):
    """Working memory + context assembly subsystem."""

    name = "prefrontal_cortex"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        config = app.config_manager.get()
        self._buffer = ConversationBuffer(
            buffer_size=config.gist.active_buffer_size,
            primacy_pins=config.gist.primacy_pins,
        )

    def add_turn(self, role: str, content: str) -> list[ConversationTurn]:
        """Add a turn to the conversation buffer.

        Returns any turns evicted from the active window (for gist compression).
        """
        return self._buffer.add_turn(role, content)

    def build_context(
        self,
        *,
        self_schema: SelfSchema | None = None,
        emotional_state: AppraisalResult | None = None,
        recalled_memories: list[dict] | None = None,
        active_episodes: list[dict] | None = None,
        temperament: dict | None = None,
    ) -> tuple[str, list[dict]]:
        """Build enriched context for the LLM.

        Returns (system_prompt, messages) ready for the LLM adapter.
        """
        system_prompt = build_system_prompt(
            self_schema=self_schema,
            emotional_state=emotional_state,
            recalled_memories=recalled_memories,
            active_episodes=active_episodes,
            temperament=temperament,
        )
        messages = build_messages(self._buffer.get_full_session())
        return system_prompt, messages

    def get_conversation_history(self) -> list[dict]:
        """Get full session history as LLM message dicts."""
        return build_messages(self._buffer.get_full_session())

    def clear(self) -> None:
        """Clear the buffer for a new session."""
        self._buffer.clear()
