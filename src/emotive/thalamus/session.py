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
                consol_result = run_consolidation(
                    consol_session, app.embedding_service, config,
                    event_bus=app.event_bus,
                    conversation_id=conv_id,
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

    # 5. Save mood state to DB
    if config.layers.mood:
        try:
            thalamus.mood.save()
            result["mood_saved"] = True
            logger.info("Mood saved at session end")
        except Exception:
            logger.exception("Mood save failed at session end")

    # 6. Publish session ended
    app.event_bus.publish(
        SESSION_ENDED,
        result,
        conversation_id=conv_id,
    )

    # 6. Clear PFC buffer + reset repetition monitor
    thalamus.prefrontal.clear()
    thalamus._repetition_monitor.reset()
    thalamus.conversation_id = None

    logger.info("Session ended: %s", conv_id)
    return result
