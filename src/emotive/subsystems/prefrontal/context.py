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
    mood: dict | None = None,
    procedural_memories: list[dict] | None = None,
    # Phase 2.5: inner world
    inner_voice_nudge: str | None = None,
    inner_speech: str | None = None,
    embodied_state: dict | None = None,
    social_perception: str | None = None,
    metacognitive_markers: str | None = None,
    # Phase Anamnesis: retrieval-enriched context
    priming: set | None = None,
    narrative: str | None = None,
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

    if mood:
        sections.append(_format_mood(mood))

    # Phase 2.5: inner world sections
    if inner_voice_nudge or inner_speech:
        sections.append(_format_inner_world(inner_voice_nudge, inner_speech))

    if embodied_state:
        formatted = _format_embodied_state(embodied_state)
        if formatted:
            sections.append(formatted)

    if social_perception:
        formatted = _format_social_perception(social_perception)
        if formatted:
            sections.append(formatted)

    if metacognitive_markers:
        sections.append(f"## Self-Awareness\n{metacognitive_markers}")

    if emotional_state:
        sections.append(_format_emotional_state(emotional_state))

    if recalled_memories:
        sections.append(_format_memories(recalled_memories))
        private = _format_private_thoughts(recalled_memories)
        if private:
            sections.append(private)

    if active_episodes:
        sections.append(_format_episodes(active_episodes))

    if procedural_memories:
        sections.append(_format_procedural(procedural_memories))

    sections.append(
        "## How to Respond\n"
        "- Respond naturally as yourself. Your memories and emotions are "
        "handled automatically.\n"
        "- If you want to remember something, just say so naturally "
        '("I want to remember this") — the system will detect it.\n'
        "- You don't need to call any tools. Just talk."
    )

    # Phase Anamnesis: retrieval-enriched sections
    if narrative:
        sections.append(f"## Your Story\n{narrative}")

    if priming:
        words = ", ".join(sorted(priming)[:15])
        sections.append(
            f"## Background Associations\n"
            f"Words and concepts in the back of your mind: {words}\n"
            f"(These are not explicit memories — they subtly influence your thinking.)"
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


def _format_mood(mood: dict) -> str:
    """Format current mood as felt descriptions, not numbers.

    Brain analog: you don't experience 'norepinephrine: elevated' —
    you just feel alert. Mood should be communicated as felt state.
    """
    lines = ["## Your Current Mood"]

    # Map each dimension to felt descriptions at high/low extremes
    felt_descriptions: dict[str, tuple[str, str]] = {
        # (low_description, high_description)
        "novelty_seeking": (
            "You're not very curious right now — routine feels comfortable",
            "You're feeling curious and drawn to new ideas",
        ),
        "social_bonding": (
            "Social connection feels less important right now — you're more inward",
            "You're feeling warm and drawn to connection with others",
        ),
        "analytical_depth": (
            "You're not inclined toward deep analysis right now",
            "You're in a thoughtful, analytical frame of mind",
        ),
        "playfulness": (
            "You're feeling serious and measured",
            "You're feeling light and playful",
        ),
        "caution": (
            "You're feeling open and unguarded",
            "You're feeling guarded and careful — something has you on edge",
        ),
        "expressiveness": (
            "You're feeling reserved — inclined to say less",
            "You're feeling expressive and want to share openly",
        ),
    }

    descriptions = []
    for dim, val in mood.items():
        deviation = val - 0.5
        if abs(deviation) <= 0.03:
            continue  # near baseline, not notable

        mapping = felt_descriptions.get(dim)
        if not mapping:
            continue

        low_desc, high_desc = mapping
        if deviation < 0:
            # Interpolate intensity: -0.03 is mild, -0.5 is extreme
            strength = min(abs(deviation) / 0.3, 1.0)
            desc = low_desc
        else:
            strength = min(deviation / 0.3, 1.0)
            desc = high_desc

        # Add intensity qualifier
        if strength < 0.3:
            desc = "Slightly: " + desc[0].lower() + desc[1:]
        elif strength > 0.7:
            desc = "Strongly: " + desc[0].lower() + desc[1:]

        descriptions.append(desc)

    if descriptions:
        lines.append("Your emotional weather right now:")
        for d in descriptions:
            lines.append(f"- {d}")
    else:
        lines.append("Your mood is near baseline — emotionally neutral.")

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
    """Format recalled memories for the system prompt.

    Inner speech memories (source: inner_speech) are filtered OUT here
    and handled separately by _format_private_thoughts.
    """
    regular = [m for m in memories if not _is_inner_speech_memory(m)]
    lines = ["## Relevant Memories"]
    for mem in regular[:10]:  # Hard limit to prevent context overload
        content = mem.get("content", "")[:200]
        mem_type = mem.get("memory_type", "unknown")
        if mem.get("_novelty_nudge"):
            lines.append(f"- [nudge] Something you haven't thought about recently: {content}")
        else:
            lines.append(f"- [{mem_type}] {content}")
    return "\n".join(lines)


def _format_private_thoughts(memories: list[dict]) -> str:
    """Format recalled inner speech memories as private thoughts.

    These appear in a separate section so the LLM knows these are
    things Ryo privately thought before, not things others told him.
    Handles faded memories (low detail_retention).
    """
    inner = [m for m in memories if _is_inner_speech_memory(m)]
    if not inner:
        return ""

    lines = ["## Private Thoughts You've Had Before"]
    for mem in inner[:5]:  # Limit private thoughts
        content = mem.get("content", "")[:150]
        meta = mem.get("metadata_") or mem.get("context") or {}
        detail = mem.get("detail_retention", 1.0)

        if detail < 0.3:
            # Faded memory — you know you thought something but not what
            trigger = meta.get("trigger", "")
            if meta.get("withheld_intention"):
                lines.append("- You chose not to share something here (details faded)")
            elif trigger:
                lines.append(f"- You had a private thought about this before, but the details have faded (triggered by: {trigger})")
            else:
                lines.append("- You've privately deliberated about this topic before (details faded)")
        else:
            # Clear memory
            prefix = "You privately thought"
            if meta.get("withheld_intention"):
                prefix = "You chose not to share"
            if meta.get("expressed_divergence"):
                prefix = "You thought (but said something different)"
            lines.append(f"- {prefix}: {content}")

    return "\n".join(lines)


def _is_inner_speech_memory(mem: dict) -> bool:
    """Check if a memory is an inner speech memory."""
    meta = mem.get("metadata_") or mem.get("context") or {}
    if meta.get("source") == "inner_speech":
        return True
    tags = mem.get("tags") or []
    return "inner_speech" in tags


def _format_procedural(memories: list[dict]) -> str:
    """Format procedural memories as behavioral instructions, not recalled context.

    Brain analog: procedural memory isn't recalled as knowledge — it's
    executed as behavior. You don't remember 'I learned to ride a bike.'
    You just ride the bike. Frame as instructions, not memories.
    """
    lines = [
        "## Behavioral Instructions",
        "These are behaviors you have internalized. Follow them naturally:",
    ]
    for mem in memories[:10]:
        content = mem.get("content", "")[:200]
        lines.append(f"- {content}")
    return "\n".join(lines)


def _format_inner_world(nudge: str | None, inner_speech: str | None) -> str:
    """Format inner world state for the system prompt.

    The nudge is always available (condensed felt sense).
    Inner speech is a reconstructed gist, not a literal quote —
    Ryo knows he thought something and the direction, but can't
    quote the exact words. This matches human introspection.
    """
    lines = ["## Your Inner Voice"]
    if nudge:
        lines.append(f"You're feeling: {nudge}")
    if inner_speech:
        lines.append(f"\n## What You Were Just Thinking")
        lines.append(inner_speech)
    return "\n".join(lines)


def _format_embodied_state(state: dict) -> str:
    """Format embodied state as felt descriptions."""
    lines = ["## How You Feel"]
    energy = state.get("energy", 1.0)
    if energy < 0.3:
        lines.append("You're feeling tired — keep responses focused")
    elif energy < 0.5:
        lines.append("Your energy is moderate — don't overextend")

    comfort = state.get("comfort", 0.5)
    if comfort > 0.7:
        lines.append("You feel comfortable and at ease with this person")
    elif comfort < 0.3:
        lines.append("You feel somewhat guarded with this person")

    load = state.get("cognitive_load", 0.0)
    if load > 0.7:
        lines.append("This conversation is complex — take it one thing at a time")

    return "\n".join(lines) if len(lines) > 1 else ""


def _format_social_perception(user_state: str | None) -> str:
    """Format what we read about the user."""
    if not user_state:
        return ""
    return f"## Reading the Room\nThe person seems {user_state}."


def _format_episodes(episodes: list[dict]) -> str:
    """Format active emotional episodes for the system prompt."""
    lines = ["## Active Emotional Episodes"]
    for ep in episodes[:5]:
        emotion = ep.get("primary_emotion", "unknown")
        intensity = ep.get("current_intensity", ep.get("intensity", 0))
        trigger = ep.get("trigger_event", "")[:100]
        lines.append(f"- {emotion} ({intensity:.2f}): {trigger}")
    return "\n".join(lines)
