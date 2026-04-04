"""Narrative reconstruction — E18.

When remembering a person or theme, construct a chronological narrative
arc instead of returning discrete results. The narrative is more than
the sum of its memories — it's meaning.

Brain analog: mPFC + DMN narrative construction.
Sources: Bergson, Matter and Memory (1896) — memory as continuous flow.
"""

from __future__ import annotations

from datetime import datetime


def construct_narrative(
    memories: list[dict],
    detected_person: str | None = None,
) -> str | None:
    """Construct a brief narrative arc from recalled memories.

    Only fires for person-focused or theme-focused retrieval
    with 3+ memories. Returns a 1-3 sentence narrative or None.

    The narrative goes into PFC context as "Your Story With [Person]."
    """
    if len(memories) < 3:
        return None

    if not detected_person and not _find_common_theme(memories):
        return None

    # Sort chronologically
    chrono = sorted(memories, key=lambda m: m.get("created_at") or datetime.min)

    # Build narrative from the arc
    subject = detected_person or _find_common_theme(memories) or "this"

    # Extract emotional arc
    emotions = [m.get("primary_emotion", "neutral") for m in chrono]
    first_emotion = emotions[0] if emotions else "neutral"
    last_emotion = emotions[-1] if emotions else "neutral"

    # Extract time span
    if chrono[0].get("created_at") and chrono[-1].get("created_at"):
        first_time = chrono[0]["created_at"]
        last_time = chrono[-1]["created_at"]
        if hasattr(first_time, 'date') and hasattr(last_time, 'date'):
            days = (last_time - first_time).days
            if days > 0:
                time_span = f"over {days} days"
            else:
                time_span = "in the same session"
        else:
            time_span = "across several conversations"
    else:
        time_span = "across several conversations"

    # Build the narrative
    memory_count = len(memories)
    first_content = chrono[0].get("content", "")[:60]
    last_content = chrono[-1].get("content", "")[:60]

    narrative = (
        f"You have {memory_count} memories about {subject} {time_span}. "
        f"It started with: \"{first_content}...\" "
    )

    if first_emotion != last_emotion:
        narrative += f"The feeling shifted from {first_emotion} to {last_emotion}."
    else:
        narrative += f"The dominant feeling throughout was {first_emotion}."

    return narrative


def _find_common_theme(memories: list[dict]) -> str | None:
    """Find a common tag across 3+ memories (excluding system tags)."""
    from collections import Counter

    SYSTEM_TAGS = {
        "episodic", "semantic", "procedural", "gist",
        "conversation_summary", "conscious_intent", "inner_speech",
        "dmn_flash", "dmn_reflection", "between_session",
    }

    tag_counts: Counter[str] = Counter()
    for m in memories:
        for tag in m.get("tags", []):
            if tag not in SYSTEM_TAGS and len(tag) > 2:
                tag_counts[tag] += 1

    if tag_counts:
        best_tag, count = tag_counts.most_common(1)[0]
        if count >= 3:
            return best_tag

    return None
