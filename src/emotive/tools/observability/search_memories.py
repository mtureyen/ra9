"""search_memories: Low-level memory search with full control."""

from __future__ import annotations

from fastmcp import Context
from sqlalchemy import select

from emotive.app_context import AppContext
from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import search_by_embedding


async def search_memories_tool(
    ctx: Context,
    query: str | None = None,
    memory_type: str | None = None,
    is_formative: bool | None = None,
    tags: list[str] | None = None,
    min_confidence: float | None = None,
    limit: int = 20,
) -> dict:
    """Low-level memory search with explicit filters.

    For research queries that need specific filter combinations.
    In Phase 0, apply_biases is always False (no mood/personality biases).

    Args:
        query: Semantic search query (optional — if None, filters only).
        memory_type: Filter by "episodic", "semantic", or "procedural".
        is_formative: Filter by formative status.
        tags: Filter by tags (any match).
        min_confidence: Minimum confidence threshold.
        limit: Max results (default 20).
    """
    app: AppContext = ctx.lifespan_context

    session = app.session_factory()
    try:
        if query:
            # Semantic search path
            embedding = app.embedding_service.embed_text(query)
            results = search_by_embedding(
                session, embedding, limit=limit, memory_type=memory_type,
            )
            # Apply additional filters in Python (pgvector doesn't support all)
            results = _apply_filters(
                results, is_formative=is_formative, tags=tags,
                min_confidence=min_confidence,
            )
        else:
            # Filter-only path (no semantic search)
            results = _filter_query(
                session, memory_type=memory_type, is_formative=is_formative,
                tags=tags, min_confidence=min_confidence, limit=limit,
            )

        memories = []
        for r in results[:limit]:
            memories.append({
                "id": str(r["id"]),
                "memory_type": r["memory_type"],
                "content": r["content"],
                "similarity": round(r.get("similarity", 0), 4) if "similarity" in r else None,
                "confidence": r.get("confidence"),
                "detail_retention": r.get("detail_retention"),
                "is_formative": r.get("is_formative", False),
                "tags": r.get("tags", []),
                "created_at": str(r.get("created_at", "")),
            })

        return {
            "status": "ok",
            "data": {
                "memories": memories,
                "count": len(memories),
                "biases_applied": False,
            },
        }

    except Exception as e:
        return {"status": "error", "error": "search_failed", "message": str(e)}
    finally:
        session.close()


def _apply_filters(
    results: list[dict],
    *,
    is_formative: bool | None,
    tags: list[str] | None,
    min_confidence: float | None,
) -> list[dict]:
    filtered = results
    if is_formative is not None:
        filtered = [r for r in filtered if r.get("is_formative") == is_formative]
    if tags:
        tag_set = set(tags)
        filtered = [r for r in filtered if tag_set & set(r.get("tags", []))]
    if min_confidence is not None:
        filtered = [r for r in filtered if (r.get("confidence") or 0) >= min_confidence]
    return filtered


def _filter_query(
    session,
    *,
    memory_type: str | None,
    is_formative: bool | None,
    tags: list[str] | None,
    min_confidence: float | None,
    limit: int,
) -> list[dict]:
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .order_by(Memory.created_at.desc())
    )
    if memory_type:
        stmt = stmt.where(Memory.memory_type == memory_type)
    if is_formative is not None:
        stmt = stmt.where(Memory.is_formative.is_(is_formative))
    if min_confidence is not None:
        stmt = stmt.where(Memory.confidence >= min_confidence)
    stmt = stmt.limit(limit)

    rows = session.execute(stmt).scalars().all()
    results = []
    for m in rows:
        entry = {
            "id": m.id,
            "memory_type": m.memory_type,
            "content": m.content,
            "confidence": m.confidence,
            "detail_retention": m.detail_retention,
            "is_formative": m.is_formative,
            "tags": m.tags or [],
            "created_at": m.created_at,
        }
        # Tag filtering in Python (Postgres array overlap is possible but simpler here)
        if tags and not (set(tags) & set(entry["tags"])):
            continue
        results.append(entry)
    return results
