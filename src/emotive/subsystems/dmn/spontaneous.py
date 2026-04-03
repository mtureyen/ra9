"""Spontaneous thought generation for the enhanced DMN.

Decides whether the DMN fires a mid-session thought and finds
cross-memory connections (topically distant but emotionally similar).

Brain analog: DMN spontaneous activation during mind-wandering.
"""

from __future__ import annotations

import random

from emotive.subsystems.amygdala.fast_pass import cosine_similarity


def should_flash(
    probability: float,
    energy: float,
    suppress_below_energy: float = 0.3,
) -> bool:
    """Decide if DMN fires mid-session. Low energy suppresses.

    Args:
        probability: Base probability of firing (0-1).
        energy: Current embodied energy level (0-1).
        suppress_below_energy: Energy threshold below which flashes are suppressed.

    Returns:
        True if DMN should fire a spontaneous thought.
    """
    if energy < suppress_below_energy:
        return False
    return random.random() < probability


def find_cross_memory_connection(
    memories: list[dict],
    embedding_service: object,
) -> tuple[dict, dict] | None:
    """Find two memories that are topically distant but emotionally similar.

    Looks for pairs with low semantic similarity (different topics) but
    the same emotion tag (similar emotional charge). These are the
    interesting connections that spark insight.

    Args:
        memories: List of memory dicts with 'embedding' and 'tags' keys.
        embedding_service: Embedding service (unused but available for future use).

    Returns:
        Tuple of (memory_a, memory_b) or None if no interesting pair found.
    """
    if len(memories) < 2:
        return None

    best_pair: tuple[dict, dict] | None = None
    best_score: float = -1.0

    for i in range(len(memories)):
        for j in range(i + 1, len(memories)):
            mem_a = memories[i]
            mem_b = memories[j]

            emb_a = mem_a.get("embedding")
            emb_b = mem_b.get("embedding")
            if not emb_a or not emb_b:
                continue

            # Semantic similarity
            semantic_sim = cosine_similarity(emb_a, emb_b)

            # Emotional similarity: same emotion tag = interesting
            tags_a = set(mem_a.get("tags", []))
            tags_b = set(mem_b.get("tags", []))
            shared_emotion = bool(tags_a & tags_b)

            # We want: low semantic similarity + shared emotion tag
            # Score: emotional_match * (1 - semantic_sim)
            if shared_emotion and semantic_sim < 0.5:
                score = 1.0 - semantic_sim
                if score > best_score:
                    best_score = score
                    best_pair = (mem_a, mem_b)

    return best_pair
