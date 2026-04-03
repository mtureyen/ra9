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

from sqlalchemy.orm import Session

from emotive.db.models.temperament import Temperament
from emotive.layers.episodes import get_active_episodes, get_current_intensity
from emotive.llm.adapter import create_adapter
from emotive.logging import get_logger
from emotive.runtime.event_bus import INPUT_RECEIVED, RESPONSE_GENERATED
from emotive.runtime.sensory_buffer import SensoryBuffer
from emotive.subsystems.amygdala import Amygdala
from emotive.subsystems.appraisal_loop import SelfAppraisal
from emotive.subsystems.association_cortex import AssociationCortex
from emotive.subsystems.dmn import DefaultModeNetwork
from emotive.subsystems.embodied import EmbodiedSubsystem
from emotive.subsystems.hippocampus import Hippocampus
from emotive.subsystems.inner_speech import InnerSpeech
from emotive.subsystems.inner_voice import InnerVoice
from emotive.subsystems.metacognition import Metacognition
from emotive.subsystems.mood import MoodSubsystem
from emotive.subsystems.predictive import PredictiveProcessor
from emotive.subsystems.prefrontal import PrefrontalCortex
from emotive.subsystems.prefrontal.buffer import compress_to_gist
from emotive.subsystems.workspace import GlobalWorkspace

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
        self.mood = MoodSubsystem(app, self._bus)
        self.llm = create_adapter(config.llm)

        # Phase 2.5: inner world subsystems
        self.embodied = EmbodiedSubsystem(app, self._bus)
        self.predictive = PredictiveProcessor(app, self._bus)
        self.workspace = GlobalWorkspace(app, self._bus)
        self.metacognition = Metacognition(app, self._bus)
        self.inner_voice = InnerVoice(app, self._bus)
        self.inner_speech = InnerSpeech(app, self._bus)
        self.self_appraisal = SelfAppraisal(app, self._bus)

        self._sensory = SensoryBuffer()
        self._conv_id: uuid.UUID | None = None
        self._last_recalled: list[dict] | None = None
        self.last_debug: dict | None = None
        self._last_tone_alignment: float = 1.0
        self._prediction_error: float = 0.5

        # ACC repetition monitor (locus coeruleus novelty nudge)
        from emotive.subsystems.hippocampus.repetition import RepetitionMonitor
        self._repetition_monitor = RepetitionMonitor()

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

            # 2c. Load procedural memories (auto-activated every exchange, like self-schema)
            # Must be loaded BEFORE session.close() — uses the same session
            procedural_memories: list[dict] = []
            try:
                procedural_memories = _load_procedural_memories(session)
            except Exception:
                logger.exception("Failed to load procedural memories")
        finally:
            session.close()

        # 2b. Mood: load current state (applies homeostasis for elapsed time)
        mood_dict = None
        if config.layers.mood:
            try:
                mood_dict = self.mood.load()
                # Mood modulates amygdala sensitivity
                sensitivity = self.mood.get_modulated_sensitivity(sensitivity)
            except Exception:
                logger.exception("Failed to load mood")

        # 3. Fast appraisal + auto-recall (both use shared embedding)
        fast_appraisal = self.amygdala.fast_pass(
            input_embedding,
            sensitivity=sensitivity,
            resilience=resilience,
            formative_threshold=config.episodes.formative_intensity_threshold,
        )
        recalled = self.association_cortex.recall(
            input_embedding, processed.text, self._conv_id,
            mood=mood_dict,
        )

        # 3b. Novelty nudge: if stuck from previous exchange, inject new stimulus
        if self._repetition_monitor.is_stuck:
            if not self._repetition_monitor.cancel_nudge(fast_appraisal.vector.novelty):
                try:
                    nudge = self._get_novelty_nudge(input_embedding, recalled)
                    if nudge:
                        recalled.append(nudge)
                        logger.info("Novelty nudge injected: %s", nudge.get("content", "")[:60])
                except Exception:
                    logger.exception("Novelty nudge failed")

        # 3c. Phase 2.5: inner world stages
        nudge = None
        inner_thought = None
        metacog = None
        workspace_output = None
        prediction_error = None
        privacy_flags = None
        trust_level = None

        if config.layers.inner_world:
            try:
                # 1. Update embodied state
                self.embodied.update(
                    emotion=fast_appraisal.primary_emotion,
                    intensity=fast_appraisal.intensity,
                    num_recalled=len(recalled),
                )

                # 2. Compute prediction error
                prediction_error = self.predictive.compute_error(input_embedding)
                self._prediction_error = prediction_error

                # 2b. Quick conflict check for workspace
                acc_conflict = 0.0
                try:
                    from emotive.subsystems.hippocampus.conflict import detect_conflict
                    conflict_session = self._app.session_factory()
                    try:
                        acc_conflict = detect_conflict(
                            conflict_session,
                            self._app.embedding_service,
                            processed.text[:500],
                            content_embedding=input_embedding,
                        )
                    finally:
                        conflict_session.close()
                except Exception:
                    pass

                # 3. Global workspace — filter signals
                workspace_output = self.workspace.broadcast(
                    recalled_memories=recalled,
                    appraisal=fast_appraisal,
                    prediction_error=prediction_error,
                    mood=mood_dict or {},
                    conflict_score=acc_conflict,
                    embodied_state=self.embodied.to_dict(),
                    config=config.workspace,
                )
                # Use workspace-filtered memories instead of raw recalled
                recalled = workspace_output.broadcast_memories

                # 4. Metacognition
                metacog = self.metacognition.evaluate(
                    recalled_memories=recalled,
                    appraisal=fast_appraisal,
                    workspace_output=workspace_output,
                )

                # 5. Condensed inner voice (always-on, no LLM)
                from emotive.memory.identity import compute_person_trust
                trust_level = "unknown"
                try:
                    iw_session = self._app.session_factory()
                    try:
                        trust_level = compute_person_trust(iw_session, "user") or "unknown"
                    finally:
                        iw_session.close()
                except Exception:
                    pass

                nudge = self.inner_voice.nudge(
                    mood=mood_dict or {},
                    trust_level=trust_level,
                    user_state=fast_appraisal.user_state,
                    metacog=metacog,
                    energy=self.embodied.energy,
                    comfort=self.embodied.comfort,
                )

                # Feed-forward tone misalignment from previous exchange
                if self._last_tone_alignment < 0.3:
                    nudge = nudge + " (last response drifted from intention)"

                # 6. Extract privacy flags from recalled memories
                privacy_flags = []
                for mem in recalled:
                    tags = mem.get("tags", [])
                    if "private" in tags or "intimate" in tags:
                        # Memory has privacy-sensitive content
                        person = next((t for t in tags if t not in {
                            "private", "intimate", "episodic", "semantic",
                            "procedural", "gist", "conversation_summary",
                        }), None)
                        if person:
                            privacy_flags.append(f"{person}'s private conversation")

                # 7. Expanded inner speech (System 2, conditional)
                if config.inner_speech.enabled:
                    inner_thought = await self.inner_speech.think(
                        nudge=nudge,
                        appraisal=fast_appraisal,
                        user_message=user_message,
                        workspace_output=workspace_output,
                        metacog=metacog,
                        llm=self.llm,
                        social_bonding=mood_dict.get("social_bonding", 0.5) if mood_dict else 0.5,
                        conflict_score=acc_conflict,
                        prediction_error=prediction_error,
                        trust_level=trust_level,
                        config=config.inner_speech,
                        privacy_flags=privacy_flags or None,
                    )
            except Exception:
                logger.exception("Inner world processing failed")

        # 4. Update buffer with FULL text, compress gists for exited turns
        exited = self.prefrontal.add_turn("user", user_message)
        if exited:
            gist = compress_to_gist(exited)
            try:
                self.hippocampus.store_gist(gist, self._conv_id)
            except Exception:
                logger.exception("Failed to store gist")

        # 5. Build context
        #    Inner speech: pass reconstructed gist, not literal text.
        #    Ryo knows he thought something and the direction, but can't
        #    quote the exact words. This matches how human introspection works.
        inner_speech_gist = None
        if inner_thought and config.layers.inner_world:
            trigger = getattr(self.inner_speech, 'last_trigger_reason', None)
            emotion = fast_appraisal.primary_emotion if fast_appraisal else "unknown"
            inner_speech_gist = (
                f"You had a private thought (triggered by {trigger or 'deliberation'}). "
                f"It leaned toward {nudge or 'presence'} in the context of {emotion}. "
                f"You can reference this if asked — but you'll be reconstructing, not quoting. "
                f"Share what feels right. Keep what doesn't."
            )

        system_prompt, messages = self.prefrontal.build_context(
            self_schema=self.dmn.current,
            emotional_state=fast_appraisal,
            recalled_memories=recalled,
            active_episodes=active_episodes,
            temperament=temperament_dict,
            mood=mood_dict,
            procedural_memories=procedural_memories if procedural_memories else None,
            inner_voice_nudge=nudge,
            inner_speech=inner_speech_gist,
            embodied_state=self.embodied.to_dict() if config.layers.inner_world else None,
            social_perception=fast_appraisal.user_state if config.layers.inner_world else None,
            metacognitive_markers=metacog.to_felt_description() if metacog else None,
        )

        # 6. Stream LLM response
        full_response: list[str] = []
        async for chunk in self.llm.stream(system_prompt, messages):
            full_response.append(chunk)
            yield chunk

        response_text = "".join(full_response)

        # === POST-PROCESSING (fire-and-forget) ===
        # Store recalled for context tag inheritance
        self._last_recalled = recalled

        debug = self._post_process(
            user_message, response_text, fast_appraisal,
            sensitivity=sensitivity, resilience=resilience,
            nudge=nudge,
            inner_thought=inner_thought,
            config=config,
            trigger_reason=getattr(self.inner_speech, 'last_trigger_reason', None),
            privacy_flags=privacy_flags if config.layers.inner_world else None,
            metacog=metacog,
            trust_level=trust_level if config.layers.inner_world else None,
        )

        # Store debug info for the terminal to display
        debug["recalled_count"] = len(recalled)
        debug["recalled_top"] = (
            recalled[0].get("content", "")[:80] if recalled else None
        )
        debug["gist_compressed"] = len(exited) if exited else 0
        if mood_dict:
            debug["mood"] = mood_dict

        # Phase 2.5: inner world debug fields
        if config.layers.inner_world:
            debug["inner_voice_nudge"] = nudge
            debug["inner_speech"] = inner_thought
            debug["social_perception"] = fast_appraisal.user_state
            debug["prediction_error"] = prediction_error
            debug["embodied_energy"] = self.embodied.energy
            debug["embodied_comfort"] = self.embodied.comfort
            if inner_thought is None and nudge is not None:
                debug["system2_bypassed"] = True

        self.last_debug = debug

    def _post_process(
        self,
        user_msg: str,
        llm_response: str,
        fast_appraisal,
        *,
        sensitivity: float = 0.5,
        resilience: float = 0.5,
        nudge: str | None = None,
        inner_thought: str | None = None,
        config=None,
        trigger_reason: str | None = None,
        privacy_flags: list[str] | None = None,
        metacog=None,
        trust_level: str | None = None,
    ) -> dict:
        """Non-blocking post-processing. Each step wrapped independently.
        One failure doesn't stop the others. Never crashes the chat.

        Returns a debug summary of what the brain did.
        """
        if config is None:
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
        #    Pass context tags from recalled memories (context inheritance)
        context_tags = []
        for mem in (self._last_recalled or []):
            for tag in mem.get("tags", []):
                if tag not in context_tags:
                    context_tags.append(tag)

        # Add inner voice nudge as context tag on exchange memory
        if nudge:
            context_tags.append(f"nudge:{nudge}")

        try:
            self.hippocampus.reset_exchange()
            # Feed prediction error into encoding threshold
            self.hippocampus._encoder.set_prediction_error(self._prediction_error)
            memory, episode_id = self.hippocampus.process_appraisal(
                final, user_msg, llm_response, self._conv_id,
                context_tags=context_tags if context_tags else None,
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

        # 5. ACC repetition check: embed response, track novelty
        try:
            response_embedding = self._app.embedding_service.embed_text(
                llm_response[:500]
            )
            stuck = self._repetition_monitor.update(
                response_embedding, fast_appraisal.vector.novelty
            )
            debug["loop_detected"] = stuck
        except Exception:
            logger.exception("Repetition monitor failed")
            debug["loop_detected"] = False

        # 6. Publish for any subscribers (future mood, etc.)
        self._bus.publish(
            RESPONSE_GENERATED,
            {
                "user_message": user_msg[:500],
                "intensity": final.intensity,
                "emotion": final.primary_emotion,
            },
            conversation_id=self._conv_id,
        )

        # 7. Phase 2.5: inner world post-processing
        if config and config.layers.inner_world:
            appraisal_result = {}  # default if self-appraisal fails
            # Self-output appraisal
            try:
                response_embedding = self._app.embedding_service.embed_text(
                    llm_response[:500]
                )
                recalled_embeddings = [
                    list(m.get("embedding", []))
                    for m in (self._last_recalled or [])
                    if m.get("embedding")
                ]
                appraisal_result = self.self_appraisal.evaluate(
                    response_text=llm_response,
                    response_embedding=response_embedding,
                    nudge=nudge or "present",
                    inner_speech=inner_thought,
                    recalled_embeddings=recalled_embeddings,
                )
                debug["tone_alignment"] = appraisal_result.get("tone_alignment")
                debug["discovery"] = appraisal_result.get("discovery")
                self._last_tone_alignment = appraisal_result.get("tone_alignment", 1.0)

                # Store self-discovery as episodic memory
                if appraisal_result.get("discovery"):
                    try:
                        from emotive.memory.episodic import store_episodic
                        disc_session = self._app.session_factory()
                        try:
                            store_episodic(
                                disc_session, self._app.embedding_service,
                                content=f"Self-discovery: {llm_response[:200]}",
                                conversation_id=self._conv_id,
                                tags=["self_discovery"],
                                context={"origin": "self_output_appraisal"},
                            )
                            disc_session.commit()
                        finally:
                            disc_session.close()
                    except Exception:
                        logger.exception("Failed to store self-discovery memory")
            except Exception:
                logger.exception("Self-appraisal failed")

            # Store inner speech as episodic memory (variable encoding)
            if inner_thought is not None:
                try:
                    from emotive.memory.episodic import store_episodic

                    # Compute encoding significance (continuous, not binary)
                    is_divergent = appraisal_result.get("tone_alignment", 1.0) < 0.3
                    is_withheld = bool(privacy_flags) if privacy_flags else False
                    is_conflict = (trigger_reason == "conflict") if trigger_reason else False

                    sig = 0.2  # base significance
                    if is_divergent:
                        sig += 0.4
                    if is_withheld:
                        sig += 0.3
                    if is_conflict:
                        sig += 0.2
                    sig += final.intensity * 0.3  # emotional boost
                    sig = min(sig, 1.0)

                    # Decay rate inversely proportional to significance
                    if sig > 0.6:
                        is_decay = 0.00003  # months
                    elif sig > 0.3:
                        is_decay = 0.00005  # weeks
                    else:
                        is_decay = 0.0002  # days

                    is_tags = ["inner_speech"]
                    if trigger_reason:
                        is_tags.append(trigger_reason)
                    if is_divergent:
                        is_tags.append("divergence")
                    if is_withheld:
                        is_tags.append("withheld")

                    is_session = self._app.session_factory()
                    try:
                        store_episodic(
                            is_session, self._app.embedding_service,
                            content=inner_thought,
                            conversation_id=self._conv_id,
                            tags=is_tags,
                            context={
                                "source": "inner_speech",
                                "elaboration": "expanded",
                                "trigger": trigger_reason,
                                "nudge": nudge,
                                "confidence_at_encoding": {
                                    "memory": getattr(metacog, 'memory_confidence', 0.5) if metacog else 0.5,
                                    "emotional": getattr(metacog, 'emotional_clarity', 0.5) if metacog else 0.5,
                                    "knowledge": getattr(metacog, 'knowledge_confidence', 0.5) if metacog else 0.5,
                                },
                                "tone_alignment": appraisal_result.get("tone_alignment", 1.0),
                                "expressed_divergence": is_divergent,
                                "withheld_intention": is_withheld,
                                "user_state_at_time": getattr(fast_appraisal, 'user_state', None) if fast_appraisal else None,
                                "trust_level": trust_level if trust_level else "unknown",
                            },
                            decay_rate=is_decay,
                        )
                        is_session.commit()
                        debug["inner_speech_stored"] = True
                        debug["inner_speech_significance"] = round(sig, 2)
                    finally:
                        is_session.close()
                except Exception:
                    logger.exception("Failed to store inner speech memory")

            # Store prediction expectation for next turn
            try:
                response_embedding = self._app.embedding_service.embed_text(
                    llm_response[:500]
                )
                self.predictive.store_expectation(response_embedding)
            except Exception:
                logger.exception("Prediction store failed")

            # DMN flash check (~5% probability)
            try:
                from emotive.subsystems.dmn.spontaneous import should_flash
                if should_flash(
                    config.dmn_enhanced.flash_probability, self.embodied.energy
                ):
                    debug["dmn_flash"] = True
            except Exception:
                logger.exception("DMN flash check failed")

        return debug


    def _get_novelty_nudge(
        self,
        input_embedding: list[float],
        already_recalled: list[dict],
    ) -> dict | None:
        """Locus coeruleus + PFC: find a relevant-but-different memory.

        Uses vector search from input but EXCLUDES already-recalled
        memories. Finds something topically connected but from a different
        cluster — a new angle on the current conversation.
        """
        already_ids = {str(m.get("id")) for m in already_recalled}

        session = self._app.session_factory()
        try:
            from emotive.db.queries.memory_queries import search_by_embedding

            candidates = search_by_embedding(
                session, input_embedding, limit=20,
            )
            # Filter out already recalled
            candidates = [
                c for c in candidates if str(c["id"]) not in already_ids
            ]
            if candidates:
                # Pick the least similar (most different but still relevant)
                candidates.sort(key=lambda c: c.get("similarity", 0))
                nudge = candidates[0]
                nudge["_novelty_nudge"] = True
                return nudge
        finally:
            session.close()
        return None


def _load_procedural_memories(session: Session) -> list[dict]:
    """Load all active procedural memories — auto-activated every exchange.

    Like the self-schema, procedural memories are always present in context.
    They represent learned behaviors that should always be active.
    """
    from sqlalchemy import select

    from emotive.db.models.memory import Memory

    stmt = (
        select(Memory)
        .where(Memory.memory_type == "procedural")
        .where(Memory.is_archived.is_(False))
        .order_by(Memory.created_at.desc())
        .limit(20)
    )
    rows = session.execute(stmt).scalars().all()
    return [
        {
            "id": str(m.id),
            "content": m.content,
            "memory_type": m.memory_type,
            "tags": m.tags or [],
        }
        for m in rows
    ]


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
