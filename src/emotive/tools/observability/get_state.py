"""get_state: Read current system state."""

from __future__ import annotations

from fastmcp import Context
from sqlalchemy import func, select

from emotive.app_context import AppContext
from emotive.db.models.memory import Memory
from emotive.db.models.temperament import Temperament


async def get_state_tool(
    ctx: Context,
    layers: list[str] | None = None,
) -> dict:
    """Read the current state of the system.

    Args:
        layers: Which layers to include. Options: "temperament", "memory_stats",
                 "episodes", "all". Defaults to ["all"].
    """
    app: AppContext = ctx.lifespan_context
    if layers is None:
        layers = ["all"]

    session = app.session_factory()
    try:
        data = {}

        if "all" in layers or "temperament" in layers:
            temp = session.get(Temperament, 1)
            if temp:
                data["temperament"] = {
                    "novelty_seeking": temp.novelty_seeking,
                    "social_bonding": temp.social_bonding,
                    "analytical_depth": temp.analytical_depth,
                    "playfulness": temp.playfulness,
                    "caution": temp.caution,
                    "expressiveness": temp.expressiveness,
                    "sensitivity": temp.sensitivity,
                    "resilience": temp.resilience,
                }

        if "all" in layers or "memory_stats" in layers:
            # Count by type
            type_counts = dict(
                session.execute(
                    select(Memory.memory_type, func.count())
                    .where(Memory.is_archived.is_(False))
                    .group_by(Memory.memory_type)
                ).all()
            )
            total = sum(type_counts.values())

            # Aggregate stats
            stats_row = session.execute(
                select(
                    func.avg(Memory.detail_retention),
                    func.count(Memory.id).filter(Memory.is_formative.is_(True)),
                    func.min(Memory.created_at),
                    func.max(Memory.created_at),
                ).where(Memory.is_archived.is_(False))
            ).one()

            data["memory_stats"] = {
                "total_memories": total,
                "by_type": type_counts,
                "average_retention": (
                    round(float(stats_row[0]), 4) if stats_row[0] else None
                ),
                "formative_count": stats_row[1] or 0,
                "oldest_memory": str(stats_row[2]) if stats_row[2] else None,
                "newest_memory": str(stats_row[3]) if stats_row[3] else None,
            }

        if "all" in layers or "episodes" in layers:
            config_check = app.config_manager.get()
            if config_check.layers.episodes:
                from emotive.layers.episodes import (
                    get_active_episodes,
                    get_current_intensity,
                )

                active = get_active_episodes(session)
                episode_data = []
                for ep in active[:5]:
                    episode_data.append({
                        "id": str(ep.id),
                        "primary_emotion": ep.primary_emotion,
                        "initial_intensity": ep.intensity,
                        "current_intensity": round(
                            get_current_intensity(ep), 4
                        ),
                        "is_formative": ep.is_formative,
                        "trigger_event": ep.trigger_event[:100],
                        "created_at": str(ep.created_at),
                    })
                data["episodes"] = {
                    "active_count": len(active),
                    "recent": episode_data,
                    "formative_count": sum(
                        1 for ep in active if ep.is_formative
                    ),
                }

        config = app.config_manager.get()
        data["active_config"] = {
            "phase": config.phase,
            "layers_enabled": [
                k for k, v in config.layers.model_dump().items() if v
            ],
        }

        return {"status": "ok", "data": data}

    except Exception as e:
        return {"status": "error", "error": "get_state_failed", "message": str(e)}
    finally:
        session.close()
