"""Basal Ganglia — reward-based memory gating (E31).

Dopamine-mediated gating determines which memories enter working memory.
Memories that were USEFUL before (led to positive conversational outcomes)
get preferential access. The system learns through reward history.

Brain analog: striatum + globus pallidus + thalamic relay.
  - "Go" pathway: phasic dopamine reinforces useful memories
  - "NoGo" pathway: dopamine dips block irrelevant memories
Sources: O'Reilly & Frank (2006), Neural Computation; PMC2440774.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.logging import get_logger

logger = get_logger("basal_ganglia")


def compute_reward_signal(
    pre_mood: dict[str, float] | None,
    post_mood: dict[str, float] | None,
    trust_change: float = 0.0,
    bonding_change: float = 0.0,
) -> float:
    """Compute reward signal from conversational outcome.

    Positive reward = trust increased, bonding improved, mood improved.
    Negative reward = trust decreased, things got worse.

    Returns reward in [-0.5, 0.5].
    """
    reward = 0.0

    # Trust and bonding changes are the strongest signal
    reward += max(-0.2, min(0.2, trust_change * 0.4))
    reward += max(-0.2, min(0.2, bonding_change * 0.4))

    # Mood improvement
    if pre_mood and post_mood:
        pre_avg = sum(pre_mood.values()) / max(len(pre_mood), 1)
        post_avg = sum(post_mood.values()) / max(len(post_mood), 1)
        mood_delta = post_avg - pre_avg
        reward += max(-0.1, min(0.1, mood_delta * 0.2))

    return max(-0.5, min(0.5, reward))


def apply_reward_to_memories(
    db_session: Session,
    memory_ids: list[uuid.UUID],
    reward: float,
) -> None:
    """Apply reward signal to retrieved memories.

    Positive reward → memory becomes easier to access next time.
    Negative reward → memory becomes slightly harder to access.
    """
    if not memory_ids or abs(reward) < 0.01:
        return

    for mid in memory_ids:
        mem = db_session.get(Memory, mid)
        if mem is None:
            continue

        meta = mem.metadata_ or {}
        current_reward = meta.get("cumulative_reward", 0.0)
        meta["cumulative_reward"] = max(-1.0, min(1.0, current_reward + reward))
        mem.metadata_ = meta

    db_session.flush()


def get_gating_bonus(memory_metadata: dict) -> float:
    """Get the reward-based gating bonus for a memory.

    Positive cumulative_reward → easier access.
    Used as an additive bonus during retrieval scoring.
    """
    reward = memory_metadata.get("cumulative_reward", 0)
    return max(0, reward) * 0.1  # up to 0.1 bonus
