"""begin_session: Start a new conversation session."""

from __future__ import annotations

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.db.models.conversation import Conversation
from emotive.db.models.temperament import Temperament
from emotive.runtime.event_bus import SESSION_STARTED


async def begin_session_tool(
    ctx: Context,
    metadata: dict | None = None,
) -> dict:
    """Start a new conversation. Returns conversation ID and current system state.

    Call this once at the start of every conversation.
    """
    app: AppContext = ctx.lifespan_context

    session = app.session_factory()
    try:
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

        config = app.config_manager.get()

        app.event_bus.publish(
            SESSION_STARTED,
            {"metadata": metadata or {}},
            conversation_id=conv.id,
        )

        session.commit()

        return {
            "status": "ok",
            "data": {
                "conversation_id": str(conv.id),
                "temperament": temperament_data,
                "active_config": {
                    "phase": config.phase,
                    "layers_enabled": [
                        k for k, v in config.layers.model_dump().items() if v
                    ],
                },
            },
        }
    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "session_start_failed", "message": str(e)}
    finally:
        session.close()
