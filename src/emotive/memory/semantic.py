"""Semantic memory: generalized patterns extracted from episodic clusters.

Phase 2.5 upgrade: LLM-generated summaries replace pipe-delimited concatenation.
The hippocampus doesn't concatenate — it extracts the gist.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import create_memory_link
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.runtime.event_bus import EventBus

from .base import store_memory

if TYPE_CHECKING:
    from emotive.llm.adapter import LLMAdapter

logger = get_logger("memory.semantic")

# Semantic decay rate: half-life ~10,000 days
SEMANTIC_DECAY_RATE = 0.00001


def store_semantic(
    session: Session,
    embedding_service: EmbeddingService,
    *,
    content: str,
    source_memory_ids: list[uuid.UUID] | None = None,
    confidence: float = 0.5,
    tags: list[str] | None = None,
    event_bus: EventBus | None = None,
) -> Memory:
    """Store a semantic memory (generalized knowledge from patterns)."""
    metadata: dict = {}
    if source_memory_ids:
        metadata["origin_memories"] = [str(mid) for mid in source_memory_ids]

    mem = store_memory(
        session,
        embedding_service,
        content=content,
        memory_type="semantic",
        tags=tags,
        metadata=metadata,
        decay_rate=SEMANTIC_DECAY_RATE,
        confidence=confidence,
        event_bus=event_bus,
    )

    # Link back to origin episodic memories
    if source_memory_ids:
        for src_id in source_memory_ids:
            create_memory_link(
                session, src_id, mem.id, "conceptual_overlap", strength=0.7
            )

    return mem


def _llm_summarize_cluster(
    llm: LLMAdapter,
    contents: list[str],
    common_tags: list[str],
) -> str | None:
    """Use the LLM to generate a semantic summary from an episodic cluster.

    Brain analog: hippocampal gist extraction — the shared meaning across
    experiences, not a concatenation of the experiences themselves.
    """
    memories_text = "\n".join(f"- {c[:200]}" for c in contents)
    tags_text = ", ".join(common_tags) if common_tags else "none"

    prompt = (
        "You are a memory consolidation system. Given these related episodic "
        "memories, extract ONE concise generalized insight — the shared meaning "
        "or pattern across them. Not a list. Not a summary of each. The gist.\n\n"
        f"Memories ({len(contents)}):\n{memories_text}\n\n"
        f"Common themes: {tags_text}\n\n"
        "Respond with ONLY the generalized insight in 1-2 sentences. "
        "No preamble, no labels, no bullet points."
    )

    try:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Inside an existing async context — create a new loop in thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    llm.generate(system="", messages=[{"role": "user", "content": prompt}]),
                )
                result = future.result(timeout=30)
        else:
            result = asyncio.run(
                llm.generate(system="", messages=[{"role": "user", "content": prompt}])
            )

        result = result.strip()
        if result and len(result) > 10:
            return result
    except Exception:
        logger.warning("LLM summarization failed, falling back to concatenation")

    return None


def _fallback_summary(contents: list[str], count: int) -> str:
    """Pipe-delimited concatenation fallback when no LLM is available."""
    return f"Pattern from {count} observations: " + " | ".join(
        c[:100] for c in contents
    )


def extract_semantic_from_cluster(
    session: Session,
    embedding_service: EmbeddingService,
    cluster_memories: list[Memory],
    *,
    event_bus: EventBus | None = None,
    llm: LLMAdapter | None = None,
) -> Memory | None:
    """Given a cluster of similar episodic memories, extract a semantic pattern.

    When an LLM adapter is provided, generates a real semantic summary —
    the shared meaning across the cluster. Falls back to concatenation
    when no LLM is available (tests, early phases).

    Brain analog: hippocampal replay consolidating episodic traces into
    generalized semantic knowledge. The gist, not the raw events.
    """
    if len(cluster_memories) < 2:
        return None

    contents = [m.content for m in cluster_memories]
    common_tags = _find_common_tags(cluster_memories)

    # Try LLM summarization, fall back to concatenation
    pattern = None
    if llm is not None:
        pattern = _llm_summarize_cluster(llm, contents, common_tags)
        if pattern:
            logger.info("LLM semantic summary: %s", pattern[:80])

    if pattern is None:
        pattern = _fallback_summary(contents, len(cluster_memories))

    source_ids = [m.id for m in cluster_memories]
    confidence = min(len(cluster_memories) / 10.0, 1.0)

    return store_semantic(
        session,
        embedding_service,
        content=pattern,
        source_memory_ids=source_ids,
        confidence=confidence,
        tags=common_tags,
        event_bus=event_bus,
    )


def _find_common_tags(memories: list[Memory]) -> list[str]:
    """Find tags that appear in at least half the memories."""
    if not memories:
        return []
    tag_counts: dict[str, int] = {}
    for m in memories:
        for tag in (m.tags or []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    threshold = len(memories) / 2
    return [tag for tag, count in tag_counts.items() if count >= threshold]
