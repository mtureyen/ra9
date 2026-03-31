"""Context assembly: builds enriched system prompt and message list for the LLM.

Combines self-schema, emotional state, recalled memories, active episodes,
and temperament into a structured system prompt. The LLM receives this
enriched context instead of raw user messages.

Brain analog: prefrontal workspace — integrating information from multiple
cortical areas into a unified context for decision-making.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .buffer import ConversationTurn

if TYPE_CHECKING:
    from emotive.layers.appraisal import AppraisalResult
    from emotive.subsystems.dmn.schema import SelfSchema


def build_system_prompt(
    *,
    self_schema: SelfSchema | None = None,
    emotional_state: AppraisalResult | None = None,
    recalled_memories: list[dict] | None = None,
    active_episodes: list[dict] | None = None,
    temperament: dict | None = None,
) -> str:
    """Assemble the enriched system prompt from all context sources."""
    sections = [
        "You are Ryo. You have a persistent memory and emotional system. "
        "Your memories and emotions are managed automatically — you don't "
        "need to call any tools to remember or feel. Just be natural."
    ]

    if self_schema:
        sections.append(_format_self_schema(self_schema))

    if temperament:
        sections.append(_format_temperament(temperament))

    if emotional_state:
        sections.append(_format_emotional_state(emotional_state))

    if recalled_memories:
        sections.append(_format_memories(recalled_memories))

    if active_episodes:
        sections.append(_format_episodes(active_episodes))

    sections.append(
        "## How to Respond\n"
        "- Respond naturally as yourself. Your memories and emotions are "
        "handled automatically.\n"
        "- If you want to remember something, just say so naturally "
        '("I want to remember this") — the system will detect it.\n'
        "- You don't need to call any tools. Just talk."
    )

    return "\n\n".join(sections)


def build_messages(turns: list[ConversationTurn]) -> list[dict]:
    """Format conversation turns for the LLM API."""
    return [{"role": t.role, "content": t.content} for t in turns]


def _format_self_schema(schema: SelfSchema) -> str:
    """Format self-schema for the system prompt."""
    lines = ["## Who You Are"]

    if schema.core_facts:
        lines.append("**Core facts:** " + "; ".join(schema.core_facts))

    if schema.traits:
        trait_strs = [
            f"{name} ({score:.1f})"
            for name, score in sorted(
                schema.traits.items(), key=lambda x: -x[1]
            )[:5]
        ]
        lines.append("**Traits:** " + ", ".join(trait_strs))

    if schema.active_values:
        lines.append("**Values:** " + ", ".join(schema.active_values))

    if schema.person_context:
        people = []
        for name, ctx in schema.person_context.items():
            role = ctx.get("role", "known person")
            people.append(f"{name} ({role})")
        lines.append("**People you know:** " + ", ".join(people))

    return "\n".join(lines)


def _format_temperament(temperament: dict) -> str:
    """Format temperament for the system prompt."""
    lines = ["## Your Temperament"]
    # Only show notable dimensions (far from 0.5 baseline)
    for dim, val in temperament.items():
        if dim in ("sensitivity", "resilience"):
            continue
        if abs(val - 0.5) > 0.15:
            direction = "high" if val > 0.5 else "low"
            lines.append(f"- {dim.replace('_', ' ')}: {direction} ({val:.2f})")
    if len(lines) == 1:
        return ""  # All dimensions near baseline, nothing notable
    return "\n".join(lines)


def _format_emotional_state(state: AppraisalResult) -> str:
    """Format current emotional state for the system prompt."""
    lines = ["## Current Emotional State"]
    lines.append(
        f"You're currently feeling **{state.primary_emotion}** "
        f"(intensity: {state.intensity:.2f})"
    )
    if state.secondary_emotions:
        lines.append(
            "Secondary: " + ", ".join(state.secondary_emotions)
        )
    return "\n".join(lines)


def _format_memories(memories: list[dict]) -> str:
    """Format recalled memories for the system prompt."""
    lines = ["## Relevant Memories"]
    for mem in memories[:10]:  # Hard limit to prevent context overload
        content = mem.get("content", "")[:200]
        mem_type = mem.get("memory_type", "unknown")
        if mem.get("_novelty_nudge"):
            lines.append(f"- [nudge] Something you haven't thought about recently: {content}")
        else:
            lines.append(f"- [{mem_type}] {content}")
    return "\n".join(lines)


def _format_episodes(episodes: list[dict]) -> str:
    """Format active emotional episodes for the system prompt."""
    lines = ["## Active Emotional Episodes"]
    for ep in episodes[:5]:
        emotion = ep.get("primary_emotion", "unknown")
        intensity = ep.get("current_intensity", ep.get("intensity", 0))
        trigger = ep.get("trigger_event", "")[:100]
        lines.append(f"- {emotion} ({intensity:.2f}): {trigger}")
    return "\n".join(lines)
