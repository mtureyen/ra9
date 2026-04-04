"""Tone alignment monitor for self-output appraisal.

Checks if the generated response's tone matches the inner voice nudge
by looking for nudge-specific keywords.

Brain analog: ACC post-response monitoring -- "did I say that right?"
"""

from __future__ import annotations


NUDGE_KEYWORDS: dict[str, list[str]] = {
    "warm": ["care", "appreciate", "glad", "thank", "love"],
    "guard": ["careful", "not sure", "won't", "can't", "no"],
    "playful": ["haha", "lol", "\U0001f604", "joke", "fun"],
    "gentle": ["understand", "hear you", "okay", "safe"],
    "cautious": ["hmm", "not sure", "careful", "maybe"],
}


def check_tone_alignment(response_text: str, nudge: str) -> float:
    """Check if response tone matches the inner voice nudge.

    Returns:
        Alignment score in [0, 1]. Higher = better match.
    """
    keywords = NUDGE_KEYWORDS.get(nudge, [])
    if not keywords:
        return 0.5  # No keywords for this nudge, neutral

    response_lower = response_text.lower()
    matches = sum(1 for kw in keywords if kw.lower() in response_lower)
    return min(matches / max(len(keywords), 1), 1.0)
