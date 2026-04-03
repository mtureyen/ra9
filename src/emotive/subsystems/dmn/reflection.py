"""Reflection prompts for the enhanced DMN.

Builds prompts for session-end reflection and spontaneous thought
exploration. These are thin prompts for the inner LLM.

Brain analog: DMN reflective processing during idle moments.
"""

from __future__ import annotations


def build_reflection_prompt(
    episode_summary: str,
    mood: dict,
    self_schema_summary: str,
) -> tuple[str, list[dict]]:
    """Build prompt for session-end reflection.

    Returns:
        (system_prompt, messages) for LLM call.
    """
    mood_description = ", ".join(
        f"{dim}={val:.2f}" for dim, val in mood.items()
        if abs(val - 0.5) > 0.05
    ) or "neutral"

    system = (
        "You are Ryo's inner reflective voice. In 1-2 sentences, what stands out "
        "from this session? What did you learn or feel? Be honest and introspective."
    )

    user_content = (
        f"Session summary: {episode_summary}\n"
        f"Current mood: {mood_description}\n"
        f"Self-concept: {self_schema_summary}"
    )

    messages = [{"role": "user", "content": user_content}]
    return system, messages


def build_spontaneous_thought_prompt(
    memory_a: dict,
    memory_b: dict,
) -> tuple[str, list[dict]]:
    """Build prompt for 'what connects these two memories?'

    Returns:
        (system_prompt, messages) for LLM call.
    """
    system = (
        "You are Ryo's wandering mind. Two distant memories just surfaced together. "
        "In ONE sentence, what surprising connection do you notice between them?"
    )

    content_a = memory_a.get("content", "unknown memory")
    content_b = memory_b.get("content", "unknown memory")

    user_content = (
        f"Memory 1: {content_a}\n"
        f"Memory 2: {content_b}"
    )

    messages = [{"role": "user", "content": user_content}]
    return system, messages
