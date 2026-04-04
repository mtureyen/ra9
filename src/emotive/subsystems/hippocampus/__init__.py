"""Hippocampus subsystem: memory encoding (unconscious + conscious intent).

Handles both encoding paths:
- Unconscious: auto-encodes when appraisal intensity > threshold
- Conscious intent: detects "I want to remember this" in LLM output

Also stores gist summaries from the PFC conversation buffer.

Brain analog: hippocampus — involuntary encoding, arousal-gated.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from emotive.db.models.memory import Memory
from emotive.layers.appraisal import AppraisalResult
from emotive.logging import get_logger
from emotive.memory.base import store_memory
from emotive.runtime.event_bus import APPRAISAL_COMPLETE, GIST_CREATED
from emotive.subsystems import Subsystem

from .encoding import UnconsciousEncoder
from .intent import detect_encoding_intent, enhanced_encode

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("hippocampus")


class Hippocampus(Subsystem):
    """Memory encoding subsystem."""

    name = "hippocampus"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        config = app.config_manager.get()
        self._encoder = UnconsciousEncoder(config.unconscious_encoding)

    def process_appraisal(
        self,
        appraisal: AppraisalResult,
        user_message: str,
        llm_response: str,
        conversation_id: uuid.UUID | None = None,
        context_tags: list[str] | None = None,
        encoding_mood: dict | None = None,
    ) -> tuple[Memory | None, uuid.UUID | None]:
        """Process a final appraisal result — encode if significant.

        context_tags: tags from co-active memories (passed from thalamus).
        These are inherited by the new memory — context inheritance.

        Returns (memory, episode_id) or (None, None).
        """
        if not self._app.config_manager.get().unconscious_encoding.enabled:
            return None, None

        config = self._app.config_manager.get()
        session = self._app.session_factory()
        try:
            # Load temperament for sensitivity/resilience
            from emotive.db.models.temperament import Temperament

            temp = session.get(Temperament, 1)
            sensitivity = temp.sensitivity if temp else 0.5
            resilience = temp.resilience if temp else 0.5

            content = f"{user_message}"
            memory, episode_id = self._encoder.encode(
                session,
                self._app.embedding_service,
                appraisal,
                content,
                conversation_id=conversation_id,
                sensitivity=sensitivity,
                resilience=resilience,
                context_tags=context_tags,
                event_bus=self._bus,
                encoding_mood=encoding_mood,
            )
            session.commit()
            return memory, episode_id
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def detect_intent(
        self,
        llm_response: str,
        conversation_id: uuid.UUID | None = None,
    ) -> bool:
        """Scan LLM response for conscious encoding intent.

        If detected, triggers enhanced encoding of the response content.
        Returns True if intent was detected and encoding occurred.
        """
        if not detect_encoding_intent(llm_response):
            return False

        session = self._app.session_factory()
        try:
            enhanced_encode(
                session,
                self._app.embedding_service,
                llm_response,
                conversation_id=conversation_id,
                event_bus=self._bus,
            )
            session.commit()
            return True
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def store_gist(
        self,
        gist_text: str,
        conversation_id: uuid.UUID | None = None,
    ) -> Memory:
        """Store a compressed conversation gist as an episodic memory.

        Gists are stored identically to other episodic memories — they get
        embeddings, instant linking, temporal linking, and decay. The only
        difference is the metadata origin tag for researcher observability.
        """
        session = self._app.session_factory()
        try:
            memory = store_memory(
                session,
                self._app.embedding_service,
                content=gist_text,
                memory_type="episodic",
                conversation_id=conversation_id,
                metadata={"origin": "gist_compression"},
                tags=["gist", "conversation_summary"],
                event_bus=self._bus,
            )
            self._bus.publish(
                GIST_CREATED,
                {"content": gist_text[:200]},
                memory_id=memory.id,
                conversation_id=conversation_id,
            )
            session.commit()
            return memory
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def reset_exchange(self) -> None:
        """Reset per-exchange encoding counters."""
        self._encoder.reset_exchange()
