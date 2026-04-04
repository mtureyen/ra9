"""Session lifecycle: staged boot and graceful shutdown.

Boot sequence mirrors the brain's waking process:
1. Load core parameters (neocortical analog — stable)
2. Load/regenerate self-model (DMN analog)
3. Load recent episodes (hippocampal reconnection)
4. Establish context (prefrontal online)
5. Ready for input

End sequence triggers consolidation and self-schema regeneration.

Brain analog: waking (staged cortical activation) and sleep
(hippocampal replay, consolidation).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from emotive.db.models.conversation import Conversation
from emotive.db.models.temperament import Temperament
from emotive.layers.episodes import archive_decayed_episodes, get_active_episodes
from emotive.logging import get_logger
from emotive.memory.consolidation import run_consolidation
from emotive.memory.session_cleanup import close_orphaned_sessions
from emotive.runtime.event_bus import SESSION_ENDED, SESSION_STARTED

if TYPE_CHECKING:
    from emotive.thalamus.dispatcher import Thalamus

logger = get_logger("thalamus.session")


def boot_session(thalamus: Thalamus) -> uuid.UUID:
    """Staged boot: initialize all subsystems and create a conversation.

    Returns the conversation ID for this session.
    """
    app = thalamus._app
    config = app.config_manager.get()

    session = app.session_factory()
    try:
        # 1. Clean orphaned sessions
        orphans = close_orphaned_sessions(
            session, app.embedding_service, config,
            event_bus=app.event_bus,
        )
        if orphans:
            logger.info("Cleaned %d orphaned sessions", orphans)

        # 2. Create conversation record
        conv = Conversation()
        session.add(conv)
        session.flush()
        conv_id = conv.id

        # 3. Regenerate self-schema (DMN activation)
        if config.self_schema.enabled:
            try:
                thalamus.dmn.regenerate()
                logger.info("Self-schema regenerated at boot")
            except Exception:
                logger.exception("DMN regeneration failed at boot")

        # 4. Load temperament baseline
        temp = session.get(Temperament, 1)
        if temp:
            logger.info("Temperament loaded (sensitivity=%.2f)", temp.sensitivity)

        # 5. Load mood state (applies homeostasis for elapsed time)
        if config.layers.mood:
            try:
                mood = thalamus.mood.load()
                shifted = {d: v for d, v in mood.items() if abs(v - 0.5) > 0.03}
                logger.info("Mood loaded: %s", shifted if shifted else "baseline")
            except Exception:
                logger.exception("Mood load failed at boot")

        # 6. Count active episodes
        active_count = 0
        if config.layers.episodes:
            active_count = len(get_active_episodes(session))
            logger.info("%d active episodes", active_count)

        app.event_bus.publish(
            SESSION_STARTED,
            {
                "orphans_cleaned": orphans,
                "active_episodes": active_count,
                "self_schema": thalamus.dmn.current is not None,
            },
            conversation_id=conv_id,
        )

        session.commit()

        # 7. Phase 2.5: inner world boot
        if config.layers.inner_world:
            try:
                thalamus.embodied.load()
                logger.info("Embodied state loaded")
            except Exception:
                logger.exception("Embodied state load failed")
            thalamus.predictive.reset()

        # 8. Phase Anamnesis: build person-node cache + reset retrieval state
        if config.layers.anamnesis:
            try:
                cache_session = app.session_factory()
                try:
                    thalamus._person_cache.build_from_schema(
                        thalamus.dmn.current, cache_session,
                    )
                    logger.info("Person-node cache built at boot")
                finally:
                    cache_session.close()
            except Exception:
                logger.exception("Person-node cache build failed")
            thalamus._retrieval_state.reset()
            thalamus._pattern_separator.reset()

            # E21: Load prospective memories (dormant intentions)
            try:
                pm_session = app.session_factory()
                try:
                    from emotive.subsystems.hippocampus.retrieval.prospective import (
                        load_prospective_memories,
                    )
                    thalamus._retrieval_state._prospective_cache = load_prospective_memories(
                        pm_session,
                    )
                    pm_session.commit()
                    if thalamus._retrieval_state._prospective_cache:
                        logger.info(
                            "Loaded %d prospective memories",
                            len(thalamus._retrieval_state._prospective_cache),
                        )
                finally:
                    pm_session.close()
            except Exception:
                logger.exception("Prospective memory load failed")
                thalamus._retrieval_state._prospective_cache = []

        # Set conversation ID on thalamus + reset repetition monitor
        thalamus.conversation_id = conv_id
        thalamus._repetition_monitor.reset()

        logger.info("Session booted: %s", conv_id)
        return conv_id

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def end_session(thalamus: Thalamus) -> dict:
    """End session: archive episodes, consolidate, regenerate self-schema.

    Returns a summary of what happened during shutdown.
    """
    app = thalamus._app
    config = app.config_manager.get()
    conv_id = thalamus.conversation_id
    result = {"conversation_id": str(conv_id) if conv_id else None}

    session = app.session_factory()
    try:
        # 1. Close conversation record
        if conv_id:
            from datetime import datetime, timezone

            conv = session.get(Conversation, conv_id)
            if conv:
                conv.ended_at = datetime.now(timezone.utc)

        # 2. Archive decayed episodes
        archived = 0
        if config.layers.episodes:
            temp = session.get(Temperament, 1)
            sensitivity = temp.sensitivity if temp else 0.5
            archived = archive_decayed_episodes(
                session, sensitivity, event_bus=app.event_bus,
            )
        result["episodes_archived"] = archived

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to close session")
    finally:
        session.close()

    # 3. Run consolidation (separate session)
    if config.consolidation.auto_on_session_end:
        try:
            consol_session = app.session_factory()
            try:
                # Pass LLM adapter for smart semantic summaries
                llm_adapter = thalamus.llm if hasattr(thalamus, 'llm') else None
                consol_result = run_consolidation(
                    consol_session, app.embedding_service, config,
                    event_bus=app.event_bus,
                    conversation_id=conv_id,
                    llm=llm_adapter,
                )
                result["consolidation"] = consol_result
                consol_session.commit()
            except Exception:
                consol_session.rollback()
                logger.exception("Consolidation failed")
            finally:
                consol_session.close()
        except Exception:
            logger.exception("Failed to create consolidation session")

    # 4. Regenerate self-schema post-consolidation
    if config.self_schema.enabled:
        try:
            thalamus.dmn.regenerate()
            result["self_schema_regenerated"] = True
        except Exception:
            logger.exception("Post-session DMN regeneration failed")
            result["self_schema_regenerated"] = False

    # 5. Between-session DMN processing
    if config.layers.inner_world:
        try:
            from emotive.subsystems.dmn.spontaneous import find_cross_memory_connection
            dmn_session = app.session_factory()
            try:
                # Load recent memories for cross-memory connection finding
                from sqlalchemy import select
                from emotive.db.models.memory import Memory
                stmt = (
                    select(Memory)
                    .where(Memory.is_archived.is_(False))
                    .order_by(Memory.created_at.desc())
                    .limit(20)
                )
                rows = dmn_session.execute(stmt).scalars().all()
                recent = [
                    {
                        "id": str(m.id),
                        "content": m.content,
                        "tags": m.tags or [],
                        "embedding": list(m.embedding) if m.embedding is not None else None,
                    }
                    for m in rows
                ]
                pair = find_cross_memory_connection(
                    recent, app.embedding_service,
                )
                if pair:
                    mem_a, mem_b = pair
                    connection = (
                        f"Between-session reflection: connection between "
                        f"'{mem_a.get('content', '?')[:80]}' and "
                        f"'{mem_b.get('content', '?')[:80]}'"
                    )
                    from emotive.memory.episodic import store_episodic
                    store_episodic(
                        dmn_session, app.embedding_service,
                        content=connection,
                        conversation_id=conv_id,
                        tags=["dmn_reflection", "between_session"],
                    )
                    dmn_session.commit()
                    result["dmn_reflection"] = True
                    logger.info("DMN between-session reflection stored")
                else:
                    result["dmn_reflection"] = False
            finally:
                dmn_session.close()
        except Exception:
            logger.exception("DMN between-session processing failed")

    # 5b. Wave 4 — session lifecycle mechanisms (E7, E29, E32, E16)
    if config.layers.anamnesis:
        try:
            wave4_session = app.session_factory()
            try:
                wave4_result = _run_wave4_lifecycle(wave4_session, conv_id)
                result.update(wave4_result)
                wave4_session.commit()
            except Exception:
                wave4_session.rollback()
                logger.exception("Wave 4 lifecycle mechanisms failed")
            finally:
                wave4_session.close()
        except Exception:
            logger.exception("Failed to create Wave 4 session")

    # 6. Save mood state to DB
    if config.layers.mood:
        try:
            thalamus.mood.save()
            result["mood_saved"] = True
            logger.info("Mood saved at session end")
        except Exception:
            logger.exception("Mood save failed at session end")

    # 7. Phase 2.5: save embodied state
    if config.layers.inner_world:
        try:
            thalamus.embodied.save()
            result["embodied_saved"] = True
            logger.info("Embodied state saved at session end")
        except Exception:
            logger.exception("Embodied state save failed at session end")

    # 7. Publish session ended
    app.event_bus.publish(
        SESSION_ENDED,
        result,
        conversation_id=conv_id,
    )

    # 8. Obsidian auto-export (if enabled)
    if config.obsidian.auto_export:
        try:
            from emotive.cli.export_obsidian import export
            vault_path = config.obsidian.vault_path
            export(vault_path)
            result["obsidian_exported"] = True
            logger.info("Obsidian auto-export completed")
        except Exception:
            logger.exception("Obsidian auto-export failed")

    # 9. Clear PFC buffer + reset repetition monitor
    thalamus.prefrontal.clear()
    thalamus._repetition_monitor.reset()
    thalamus.conversation_id = None

    logger.info("Session ended: %s", conv_id)
    return result


def _run_wave4_lifecycle(
    session,
    conversation_id: uuid.UUID | None,
) -> dict:
    """Wave 4: session lifecycle mechanisms.

    Runs after consolidation. Implements:
      E7:  Sequential replay — strengthen temporal links
      E29: Replay interleaving — cross-session theme connections
      E32: Between-session emotional processing (depotentiation)
      E16: Active forgetting — release irrelevant memories

    Brain analog: hippocampal replay during sleep/rest,
    DMN-mediated consolidation, active forgetting via prefrontal inhibition.
    """
    from datetime import datetime, timedelta, timezone

    from sqlalchemy import select

    from emotive.db.models.memory import Memory
    from emotive.db.queries.memory_queries import create_memory_link, strengthen_link

    now = datetime.now(timezone.utc)
    result = {}

    # --- E7: Session-end sequential replay ---
    # Get this session's memories in chronological order
    session_memories = []
    if conversation_id:
        stmt = (
            select(Memory)
            .where(Memory.conversation_id == conversation_id)
            .where(Memory.is_archived.is_(False))
            .order_by(Memory.created_at.asc())
        )
        session_memories = list(session.execute(stmt).scalars().all())

    replay_links = 0
    if len(session_memories) >= 2:
        # Forward replay: strengthen temporal links between consecutive memories
        for i in range(len(session_memories) - 1):
            strengthen_link(
                session,
                session_memories[i].id,
                session_memories[i + 1].id,
                "temporal_sequence",
                boost=0.1,
            )
            replay_links += 1

        # Reverse replay for last 5 memories (recency prioritized)
        tail = session_memories[-5:]
        for i in range(len(tail) - 1, 0, -1):
            strengthen_link(
                session,
                tail[i].id,
                tail[i - 1].id,
                "temporal_sequence",
                boost=0.05,
            )
            replay_links += 1

    result["replay_links"] = replay_links

    # --- E29: Replay interleaving ---
    # Find old memories from different sessions that share tags with today's
    interleave_links = 0
    if session_memories:
        session_tags: set[str] = set()
        session_ids: set[uuid.UUID] = set()
        for m in session_memories:
            session_tags.update(m.tags or [])
            session_ids.add(m.id)

        # Exclude generic tags
        _GENERIC_TAGS = {"conversation_summary", "gist", "conscious_intent", "meta_memory",
                         "retrieval_experience", "dmn_reflection", "between_session"}
        theme_tags = session_tags - _GENERIC_TAGS

        if theme_tags:
            from sqlalchemy import func as sa_func

            # Find old memories sharing tags but from different sessions
            tag_list = list(theme_tags)
            stmt = (
                select(Memory)
                .where(Memory.is_archived.is_(False))
                .where(Memory.tags.overlap(tag_list))
                .where(Memory.conversation_id != conversation_id)
                .order_by(sa_func.random())
                .limit(5 * min(len(theme_tags), 3))  # sample pool
            )
            old_memories = list(session.execute(stmt).scalars().all())

            # Create interleave links between session memories and old ones
            sample = old_memories[:15] if len(old_memories) > 15 else old_memories
            for old_mem in sample:
                # Find the session memory with best tag overlap
                best_match = max(
                    session_memories,
                    key=lambda sm: len(set(sm.tags or []) & set(old_mem.tags or [])),
                )
                if old_mem.id != best_match.id:
                    create_memory_link(
                        session,
                        best_match.id,
                        old_mem.id,
                        "replay_interleave",
                        strength=0.08,
                    )
                    interleave_links += 1

    result["interleave_links"] = interleave_links

    # --- E32: Between-session emotional processing ---
    # For recent emotional memories (< 48h), apply emotional depotentiation
    cutoff_48h = now - timedelta(hours=48)
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.created_at > cutoff_48h)
        .where(Memory.emotional_intensity > 0.2)
    )
    recent_emotional = list(session.execute(stmt).scalars().all())

    depotentiated = 0
    for mem in recent_emotional:
        # Exempt: formative and flashbulb (high decay_protection)
        if mem.is_formative:
            continue
        if (mem.decay_protection or 1.0) < 0.3:
            continue  # flashbulb — exempt

        # Recency factor: 1.0 at creation, 0.0 at 48h
        age_hours = (now - mem.created_at).total_seconds() / 3600
        recency_factor = max(0.0, 1.0 - age_hours / 48.0)

        ei = mem.emotional_intensity or 0.0
        new_ei = ei * (1 - 0.05 * recency_factor)
        new_ei = max(new_ei, 0.2)  # floor

        if new_ei != ei:
            mem.emotional_intensity = new_ei
            depotentiated += 1

    result["emotional_depotentiated"] = depotentiated

    # --- E16: Active forgetting ---
    # Identify release candidates: high retrieval count but declining relevance
    cutoff_7d = now - timedelta(days=7)
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.is_formative.is_(False))
        .where(Memory.retrieval_count > 10)
        .where(Memory.emotional_intensity < 0.4)
    )
    candidates = list(session.execute(stmt).scalars().all())

    released = 0
    for mem in candidates:
        # Must not have been retrieved in the last 7 days
        if mem.last_retrieved and mem.last_retrieved > cutoff_7d:
            continue

        # Triple decay rate
        mem.decay_rate = (mem.decay_rate or 0.0001) * 3.0
        # Reduce activation to floor
        mem.current_activation = 0.3  # hard to find but not impossible
        # Tag as releasing
        tags = list(mem.tags or [])
        if "releasing" not in tags:
            tags.append("releasing")
            mem.tags = tags

        released += 1

    result["active_forgetting_released"] = released

    session.flush()
    logger.info(
        "Wave 4 lifecycle: replay_links=%d, interleave=%d, depotentiated=%d, released=%d",
        replay_links, interleave_links, depotentiated, released,
    )
    return result
