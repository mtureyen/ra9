"""begin_session: Start a new conversation session."""

from __future__ import annotations

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.conversation import Conversation
from emotive.db.models.memory import Memory
from emotive.db.models.temperament import Temperament
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
        identity_memories = _load_identity_memories(session)

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


def _load_identity_memories(session, limit: int = 10) -> list[dict]:
    """Load the most important memories for identity continuity.

    Pulls memories by: highest retrieval count, highest significance,
    and formative status. These are the memories that define who you are.
    """
    # Most retrieved (identity anchors — what gets recalled every session)
    most_retrieved = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.retrieval_count > 0)
        .order_by(Memory.retrieval_count.desc())
        .limit(5)
    )

    # Highest significance
    high_sig = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.metadata_["significance"].as_float() >= 0.8)
        .order_by(Memory.metadata_["significance"].as_float().desc())
        .limit(5)
    )

    # Formative memories
    formative = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.is_formative.is_(True))
        .limit(5)
    )

    # Combine and deduplicate
    seen_ids = set()
    results = []

    for stmt in [most_retrieved, high_sig, formative]:
        rows = session.execute(stmt).scalars().all()
        for m in rows:
            if m.id not in seen_ids and len(results) < limit:
                seen_ids.add(m.id)
                sig = None
                if m.metadata_ and "significance" in m.metadata_:
                    sig = m.metadata_["significance"]
                results.append({
                    "id": str(m.id),
                    "memory_type": m.memory_type,
                    "content": m.content,
                    "significance": sig,
                    "retrieval_count": m.retrieval_count,
                    "is_formative": m.is_formative,
                    "tags": m.tags or [],
                })

    return results
