"""Prompt construction for expanded inner speech.

Builds a thin prompt for the inner LLM when System 2 gate opens.
The prompt includes the nudge, emotion, trust level, and user state.

Brain analog: dlPFC deliberative processing -- "let me think about this."
"""

from __future__ import annotations


def build_inner_speech_prompt(
    nudge: str,
    emotion: str,
    intensity: float,
    user_state: str | None,
    user_message: str,
    trust_level: str,
    privacy_flags: list[str] | None = None,
) -> tuple[str, list[dict]]:
    """Build thin prompt for expanded inner speech.

    Returns:
        (system_prompt, messages) for LLM call.
    """
    system = (
        "You are Ryo's inner voice. In ONE sentence, what should guide your response? "
        f"Current feeling: {nudge}. Emotion: {emotion} ({intensity:.1f}). "
        f"This person is {trust_level}."
    )
    if user_state:
        system += f" They seem {user_state}."
    if privacy_flags:
        system += f" Privacy: {', '.join(privacy_flags)}."

    messages = [{"role": "user", "content": f"They said: {user_message}"}]
    return system, messages
