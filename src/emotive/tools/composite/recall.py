"""recall: Full memory retrieval pipeline."""

from __future__ import annotations

import uuid as _uuid

from fastmcp import Context

from emotive.app_context import AppContext
from emotive.memory.base import recall_memories


async def recall_tool(
    ctx: Context,
    query: str,
    memory_types: list[str] | None = None,
    limit: int = 5,
    include_spreading: bool = True,
    conversation_id: str | None = None,
) -> dict:
    """Retrieve memories relevant to a query.

    Runs the full retrieval pipeline: semantic search, recency weighting,
    and spreading activation. In Phase 0, no mood or personality biases
    are applied.

    Args:
        query: What to search for.
        memory_types: Filter by type ("episodic", "semantic", "procedural").
        limit: Max memories to return (default 5).
        include_spreading: Follow associative links (default True).
    """
    app: AppContext = ctx.lifespan_context
    config = app.config_manager.get()
    weights = config.retrieval_weights
    spread = config.spreading_activation

    session = app.session_factory()
    try:
        # Phase 0: query one type at a time if filtered, else all
        memory_type = memory_types[0] if memory_types and len(memory_types) == 1 else None

        results = recall_memories(
            session,
            app.embedding_service,
            query=query,
            memory_type=memory_type,
            limit=limit,
            include_spreading=include_spreading,
            w_semantic=weights.semantic,
            w_recency=weights.recency,
            w_activation=weights.spreading_activation,
            w_significance=weights.significance,
            spreading_hops=spread.hops,
            spreading_decay=spread.decay_per_hop,
            event_bus=app.event_bus,
            conversation_id=_uuid.UUID(conversation_id) if conversation_id else None,
        )

        session.commit()

        memories = []
        for r in results:
            memories.append({
                "id": str(r["id"]),
                "memory_type": r["memory_type"],
                "content": r["content"],
                "relevance_scores": {
                    "semantic_similarity": round(r.get("similarity", 0), 4),
                    "recency_weight": round(r.get("recency_weight", 0), 4),
                    "spreading_activation": round(r.get("spreading_activation", 0), 4),
                    "significance": round(r.get("significance", 0.5), 4),
                    "final_rank": round(r.get("final_rank", 0), 4),
                },
                "is_formative": r.get("is_formative", False),
                "retrieval_count": r.get("retrieval_count", 0),
                "created_at": str(r.get("created_at", "")),
            })

        return {
            "status": "ok",
            "data": {
                "memories": memories,
                "biases_applied": {
                    "mood_congruence": "disabled (Phase 0)",
                    "personality_bias": "disabled (Phase 0)",
                },
                "total_candidates_searched": len(results),
                "query_embedding_generated": True,
            },
        }

    except Exception as e:
        session.rollback()
        return {"status": "error", "error": "recall_failed", "message": str(e)}
    finally:
        session.close()
