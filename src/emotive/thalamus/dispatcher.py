"""Thalamus: the orchestrator that routes input through subsystems.

Manages the critical path (pre-LLM) and fire-and-forget post-processing.
Each subsystem is called directly for the critical path (like neural tracts).
The EventBus carries modulatory signals for post-processing.

Brain analog: thalamus — sensory relay, routes input to cortical areas.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from emotive.db.models.temperament import Temperament
from emotive.layers.episodes import get_active_episodes, get_current_intensity
from emotive.llm.adapter import create_adapter
from emotive.logging import get_logger
from emotive.runtime.event_bus import INPUT_RECEIVED, RESPONSE_GENERATED
from emotive.runtime.sensory_buffer import SensoryBuffer
from emotive.subsystems.amygdala import Amygdala
from emotive.subsystems.association_cortex import AssociationCortex
from emotive.subsystems.dmn import DefaultModeNetwork
from emotive.subsystems.hippocampus import Hippocampus
from emotive.subsystems.prefrontal import PrefrontalCortex
from emotive.subsystems.prefrontal.buffer import compress_to_gist

if TYPE_CHECKING:
    from emotive.app_context import AppContext

logger = get_logger("thalamus")


class Thalamus:
    """The orchestrator. Routes input through subsystems,
    manages the critical path, coordinates responses."""

    def __init__(self, app: AppContext) -> None:
        self._app = app
        self._bus = app.event_bus
        config = app.config_manager.get()

        # Initialize all subsystems
        self.amygdala = Amygdala(app, self._bus)
        self.association_cortex = AssociationCortex(app, self._bus)
        self.prefrontal = PrefrontalCortex(app, self._bus)
        self.hippocampus = Hippocampus(app, self._bus)
        self.dmn = DefaultModeNetwork(app, self._bus)
        self.llm = create_adapter(config.llm)

        self._sensory = SensoryBuffer()
        self._conv_id: uuid.UUID | None = None
        self.last_debug: dict | None = None

    @property
    def conversation_id(self) -> uuid.UUID | None:
        return self._conv_id

    @conversation_id.setter
    def conversation_id(self, value: uuid.UUID | None) -> None:
        self._conv_id = value

    async def process_input(self, user_message: str) -> AsyncIterator[str]:
        """Full cycle: pre-process → LLM stream → post-process.

        The critical path (pre-LLM) blocks until context is ready.
        Post-processing is fire-and-forget — failures logged, never crash.

        Yields text chunks from the LLM. After completion, self.last_debug
        contains the brain activity summary for the exchange.
        """
        # === CRITICAL PATH (blocking) ===

        # 0. Sensory preprocessing (truncates for embedding model)
        processed = self._sensory.process(user_message)

        self._bus.publish(
            INPUT_RECEIVED,
            {"char_count": processed.char_count, "truncated": processed.truncated},
            conversation_id=self._conv_id,
        )

        # 1. Embed ONCE — shared by amygdala + association cortex
        input_embedding = self._app.embedding_service.embed_text(processed.text)

        # 2. Load temperament for appraisal params
        session = self._app.session_factory()
        try:
            temp = session.get(Temperament, 1)
            sensitivity = temp.sensitivity if temp else 0.5
            resilience = temp.resilience if temp else 0.5
            temperament_dict = _temperament_to_dict(temp) if temp else None

            # Get active episodes
            config = self._app.config_manager.get()
            active_episodes = []
            if config.layers.episodes:
                episodes = get_active_episodes(session)
                active_episodes = [
                    {
                        "primary_emotion": ep.primary_emotion,
                        "intensity": ep.intensity,
                        "current_intensity": get_current_intensity(ep),
                        "trigger_event": ep.trigger_event,
                    }
                    for ep in episodes
                ]
        finally:
            session.close()

        # 3. Fast appraisal + auto-recall (both use shared embedding)
        fast_appraisal = self.amygdala.fast_pass(
            input_embedding,
            sensitivity=sensitivity,
            resilience=resilience,
            formative_threshold=config.episodes.formative_intensity_threshold,
        )
        recalled = self.association_cortex.recall(
            input_embedding, processed.text, self._conv_id
        )

        # 4. Update buffer with FULL text, compress gists for exited turns
        exited = self.prefrontal.add_turn("user", user_message)
        if exited:
            gist = compress_to_gist(exited)
            try:
                self.hippocampus.store_gist(gist, self._conv_id)
            except Exception:
                logger.exception("Failed to store gist")

        # 5. Build context
        system_prompt, messages = self.prefrontal.build_context(
            self_schema=self.dmn.current,
            emotional_state=fast_appraisal,
            recalled_memories=recalled,
            active_episodes=active_episodes,
            temperament=temperament_dict,
        )

        # 6. Stream LLM response
        full_response: list[str] = []
        async for chunk in self.llm.stream(system_prompt, messages):
            full_response.append(chunk)
            yield chunk

        response_text = "".join(full_response)

        # === POST-PROCESSING (fire-and-forget) ===
        debug = self._post_process(
            user_message, response_text, fast_appraisal,
            sensitivity=sensitivity, resilience=resilience,
        )

        # Store debug info for the terminal to display
        debug["recalled_count"] = len(recalled)
        debug["recalled_top"] = (
            recalled[0].get("content", "")[:80] if recalled else None
        )
        debug["gist_compressed"] = len(exited) if exited else 0
        self.last_debug = debug

    def _post_process(
        self,
        user_msg: str,
        llm_response: str,
        fast_appraisal,
        *,
        sensitivity: float = 0.5,
        resilience: float = 0.5,
    ) -> dict:
        """Non-blocking post-processing. Each step wrapped independently.
        One failure doesn't stop the others. Never crashes the chat.

        Returns a debug summary of what the brain did.
        """
        config = self._app.config_manager.get()
        debug = {
            "fast_emotion": fast_appraisal.primary_emotion,
            "fast_intensity": fast_appraisal.intensity,
            "reappraised": False,
            "final_emotion": fast_appraisal.primary_emotion,
            "final_intensity": fast_appraisal.intensity,
            "encoded": False,
            "episode_id": None,
            "intent_detected": False,
        }

        # 1. Slow appraisal (can override fast pass)
        final = fast_appraisal
        try:
            final = self.amygdala.slow_pass(
                user_msg, llm_response, fast_appraisal,
                sensitivity=sensitivity,
                resilience=resilience,
                formative_threshold=config.episodes.formative_intensity_threshold,
            )
            debug["reappraised"] = final is not fast_appraisal
            debug["final_emotion"] = final.primary_emotion
            debug["final_intensity"] = final.intensity
        except Exception:
            logger.exception("Amygdala slow pass failed, using fast pass")

        # 2. Hippocampus: reset exchange counter + episode creation + encoding
        try:
            self.hippocampus.reset_exchange()
            memory, episode_id = self.hippocampus.process_appraisal(
                final, user_msg, llm_response, self._conv_id
            )
            if memory is not None:
                debug["encoded"] = True
                debug["episode_id"] = str(episode_id)
        except Exception:
            logger.exception("Hippocampus encoding failed")

        # 3. Hippocampus: conscious intent detection
        try:
            debug["intent_detected"] = self.hippocampus.detect_intent(
                llm_response, self._conv_id
            )
        except Exception:
            logger.exception("Intent detection failed")

        # 4. PFC: add assistant turn to buffer
        try:
            self.prefrontal.add_turn("assistant", llm_response)
        except Exception:
            logger.exception("PFC buffer update failed")

        # 5. Publish for any subscribers (future mood, etc.)
        self._bus.publish(
            RESPONSE_GENERATED,
            {
                "user_message": user_msg[:500],
                "intensity": final.intensity,
                "emotion": final.primary_emotion,
            },
            conversation_id=self._conv_id,
        )

        return debug


def _temperament_to_dict(temp: Temperament) -> dict:
    """Convert Temperament model to dict for context building."""
    return {
        "novelty_seeking": temp.novelty_seeking,
        "social_bonding": temp.social_bonding,
        "analytical_depth": temp.analytical_depth,
        "playfulness": temp.playfulness,
        "caution": temp.caution,
        "expressiveness": temp.expressiveness,
        "sensitivity": temp.sensitivity,
        "resilience": temp.resilience,
    }
