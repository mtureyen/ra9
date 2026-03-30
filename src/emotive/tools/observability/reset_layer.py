"""reset_layer: Reset a specific layer to its initial state."""

from __future__ import annotations

from datetime import datetime, timezone

from fastmcp import Context
from sqlalchemy import update

from emotive.app_context import AppContext
from emotive.db.models.episode import EmotionalEpisode


async def reset_layer_tool(
    ctx: Context,
    layer: str,
    reason: str,
    preserve_history: bool = True,
) -> dict:
    """Reset a specific layer to its initial state.

    For comparative experiments. Temperament cannot be reset (immutable).
    Memory cannot be reset through this tool.

    Args:
        layer: "episodes", "mood", "personality", or "identity".
        reason: Why are you resetting? (required for research traceability).
        preserve_history: Keep the history/drift logs (default True).
    """
    app: AppContext = ctx.lifespan_context

    valid_layers = {"episodes", "mood", "personality", "identity"}
    if layer not in valid_layers:
        return {
            "status": "error",
            "error": "invalid_layer",
            "message": f"Cannot reset '{layer}'. Valid: {valid_layers}",
        }

    session = app.session_factory()
    try:
        if layer == "episodes":
            # Archive all active episodes
            now = datetime.now(timezone.utc)
            result = session.execute(
                update(EmotionalEpisode)
                .where(EmotionalEpisode.is_active.is_(True))
                .values(is_active=False, archived_at=now)
            )
            count = result.rowcount

            from emotive.config.audit import audit_config_change

            audit_config_change(
                session, f"reset_layer.{layer}", "active", "reset",
                reason=reason,
            )
            session.commit()

            return {
                "status": "ok",
                "data": {
                    "layer": layer,
                    "episodes_archived": count,
                    "reason": reason,
                    "history_preserved": preserve_history,
                },
            }
        else:
            # Mood, personality, identity — tables don't exist yet
            return {
                "status": "error",
                "error": "layer_not_available",
                "message": f"Layer '{layer}' is not active in current phase",
            }

    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "reset_failed", "message": str(e)}
    finally:
        session.close()
