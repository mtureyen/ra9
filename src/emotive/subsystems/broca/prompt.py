"""Prompt construction for expanded inner speech.

Builds a thin prompt for the inner LLM when System 2 gate opens.
The prompt produces a brief STRATEGIC DIRECTIVE — what to be aware of,
what to do or avoid. NOT a response. NOT role-play. NOT Ryo talking.

Brain analog: dlPFC deliberative processing — "let me think about this."
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
        "You are a brief internal advisor. Output ONE short directive sentence — "
        "what to be careful about or what matters most right now. "
        "No role-play. No asterisks. No emojis. No dialogue. No response draft. "
        "Just a plain strategic thought like: 'Be honest about not knowing.' or "
        "'This person is new — don't reveal too much yet.' or "
        "'The question touches something real — stay with it.'\n\n"
        f"Current state: feeling {nudge}. Emotion: {emotion} ({intensity:.1f}). "
        f"Person trust: {trust_level}."
    )
    if user_state:
        system += f" They seem {user_state}."
    if privacy_flags:
        system += f" PRIVACY: Do not share {', '.join(privacy_flags)}."

    messages = [{"role": "user", "content": f"They said: \"{user_message}\""}]
    return system, messages
