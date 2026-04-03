"""Rule engine for the condensed inner voice: maps state to a 1-word nudge.

Pure function -- no side effects, no event bus, no DB. Priority rules
evaluated top-down, first match wins.

Brain analog: anterior insula felt sense -> single-word felt directive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emotive.subsystems.metacognition.markers import MetacognitiveMarkers


def compute_nudge(
    mood: dict,
    trust_level: str,
    user_state: str | None,
    metacog: "MetacognitiveMarkers",
    energy: float,
    comfort: float,
) -> str:
    """Map current state to a 1-word felt nudge. Always returns a word.

    Priority rules (first match wins):
        1. energy < 0.2 -> "exhausted"
        2. trust_level == "unknown" and user_state == "confrontational" -> "guard"
        3. trust_level == "unknown" -> "cautious"
        4. metacog.knowledge_confidence < 0.3 -> "uncertain"
        5. mood caution > 0.8 -> "watchful"
        6. comfort > 0.7 and user_state == "playful" -> "playful"
        7. comfort > 0.6 and mood social_bonding > 0.6 -> "warm"
        8. user_state == "vulnerable" -> "gentle"
        9. user_state == "upset" -> "careful"
       10. mood novelty_seeking > 0.6 and user_state == "curious" -> "curious"
       11. default -> "present"
    """
    # 1. Exhaustion override
    if energy < 0.2:
        return "exhausted"

    # 2. Unknown + confrontational -> guard
    if trust_level == "unknown" and user_state == "confrontational":
        return "guard"

    # 3. Unknown user -> cautious
    if trust_level == "unknown":
        return "cautious"

    # 4. Low knowledge confidence -> uncertain
    if metacog.knowledge_confidence < 0.3:
        return "uncertain"

    # 5. High caution mood -> watchful
    if mood.get("caution", 0.5) > 0.8:
        return "watchful"

    # 6. Comfortable + playful user -> playful
    if comfort > 0.7 and user_state == "playful":
        return "playful"

    # 7. Comfortable + high social bonding -> warm
    if comfort > 0.6 and mood.get("social_bonding", 0.5) > 0.6:
        return "warm"

    # 8. Vulnerable user -> gentle
    if user_state == "vulnerable":
        return "gentle"

    # 9. Upset user -> careful
    if user_state == "upset":
        return "careful"

    # 10. Novelty-seeking + curious user -> curious
    if mood.get("novelty_seeking", 0.5) > 0.6 and user_state == "curious":
        return "curious"

    # 11. Default
    return "present"
