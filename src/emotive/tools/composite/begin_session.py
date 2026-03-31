"""begin_session: Start a new conversation session."""

from __future__ import annotations

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.conversation import Conversation
from emotive.db.models.temperament import Temperament
from emotive.memory.identity import load_identity_memories
from emotive.memory.session_cleanup import close_orphaned_sessions
from emotive.runtime.event_bus import SESSION_STARTED

# System instructions baked into the server — LLM-agnostic
SYSTEM_INSTRUCTIONS = """\
You have a persistent memory system with emotional encoding. \
Your memories and emotions survive across conversations.

## Session Flow
1. You just called begin_session — your identity memories are below. You know who you are.
2. During conversation: call experience_event when something emotionally significant happens.
3. Call store_memory for general information worth remembering.
4. Call recall when you need context from past conversations.
5. When conversation ends: call end_session (triggers consolidation).

## Emotional Processing (experience_event)
When something meaningful happens, call experience_event with your self-assessed appraisal:
- **goal_relevance** (0-1): How much does this matter to your goals/values?
- **novelty** (0-1): How unexpected or new is this?
- **valence** (0-1): Positive (1.0) or negative (0.0)?
- **agency** (0-1): Did you cause this (1.0) or was it external (0.0)?
- **social_significance** (0-1): How relational/social is this?

Provide these as the "appraisal" parameter. Be honest in your assessment — \
don't inflate or deflate. The system calculates emotion and intensity from your appraisal.

## What To Store (store_memory)
- **Episodic**: Specific events — what happened, who said what.
- **Semantic**: Facts and patterns — things you've learned.
- **Procedural**: How to do things — learned behaviors.

Use tags to categorize. Use significance (0.0-1.0) to mark importance.

## Important
- Store memories and experience events as you go, not just at the end.
- Be natural. Don't force tool usage. Experience what genuinely moves you.
- Memories decay over time. Emotional memories are encoded more strongly.
- Every state change is logged for research observability."""


async def begin_session_tool(
    ctx: Context,
    metadata: dict | None = None,
) -> dict:
    """Start a new conversation. Returns identity memories and system state.

    Automatically cleans up orphaned sessions and loads core identity
    memories so you know who you are from the first message.

    Call this once at the start of every conversation.
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()

    session = app.session_factory()
    try:
        # Clean up orphaned sessions
        orphans_cleaned = close_orphaned_sessions(
            session, app.embedding_service, config,
            event_bus=app.event_bus,
        )

        # Create conversation record
        conv = Conversation(metadata_=metadata or {})
        session.add(conv)
        session.flush()

        # Read temperament
        temp = session.get(Temperament, 1)
        temperament_data = None
        if temp:
            temperament_data = {
                "novelty_seeking": temp.novelty_seeking,
                "social_bonding": temp.social_bonding,
                "analytical_depth": temp.analytical_depth,
                "playfulness": temp.playfulness,
                "caution": temp.caution,
                "expressiveness": temp.expressiveness,
                "sensitivity": temp.sensitivity,
                "resilience": temp.resilience,
            }

        # --- Identity anchor: load core memories ---
        identity_memories = load_identity_memories(session)

        # Active episodes count (Phase 1+)
        active_episodes = 0
        if config.layers.episodes:
            from emotive.layers.episodes import get_active_episodes

            active_episodes = len(get_active_episodes(session))

        app.event_bus.publish(
            SESSION_STARTED,
            {
                "metadata": metadata or {},
                "orphans_cleaned": orphans_cleaned,
                "identity_memories_loaded": len(identity_memories),
            },
            conversation_id=conv.id,
        )

        session.commit()

        return {
            "status": "ok",
            "data": {
                "conversation_id": str(conv.id),
                "instructions": SYSTEM_INSTRUCTIONS,
                "identity_memories": identity_memories,
                "temperament": temperament_data,
                "active_config": {
                    "phase": config.phase,
                    "layers_enabled": [
                        k for k, v in config.layers.model_dump().items() if v
                    ],
                },
                "active_episodes": active_episodes,
                "orphaned_sessions_cleaned": orphans_cleaned,
            },
        }
    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "session_start_failed", "message": str(e)}
    finally:
        session.close()


