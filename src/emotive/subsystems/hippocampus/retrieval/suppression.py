"""Intentional suppression — E13.

dlPFC and vlPFC actively down-regulate hippocampal activation during
suppression. Inner speech can produce suppression directives that
reduce memory accessibility. Suppression weakens under stress.

Brain analog: dorsolateral/ventrolateral PFC → hippocampal inhibition.
Sources: Anderson & Green (2001), PNAS.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.logging import get_logger

logger = get_logger("hippocampus.retrieval.suppression")


def apply_suppression(
    db_session: Session,
    memory_id: uuid.UUID,
    suppression_strength: float = 0.4,
) -> None:
    """Apply intentional suppression to a memory.

    Called when inner speech produces a suppression directive
    like "don't think about X" or "avoid recalling Y."
    """
    mem = db_session.get(Memory, memory_id)
    if mem is None:
        return

    current = mem.suppression_level or 0.0
    mem.suppression_level = min(current + suppression_strength, 1.0)
    mem.suppression_decay_start = datetime.now(timezone.utc)
    db_session.flush()
    logger.info("Suppression applied to %s: level=%.2f", memory_id, mem.suppression_level)


def get_effective_suppression(
    suppression_level: float,
    decay_start: datetime | None,
    energy: float = 1.0,
    arousal: float = 0.0,
    now: datetime | None = None,
) -> float:
    """Compute current effective suppression after decay and stress modulation.

    Suppression decays over 24h. Under stress (low energy, high arousal),
    suppression effectiveness is halved — defenses lower.

    Returns effective suppression level [0, 1].
    """
    if suppression_level <= 0 or decay_start is None:
        return 0.0

    if now is None:
        now = datetime.now(timezone.utc)

    # Exponential decay: τ = 24 hours
    elapsed_hours = (now - decay_start).total_seconds() / 3600
    decayed = suppression_level * math.exp(-elapsed_hours / 24.0)

    # Stress weakens suppression (E14 somatic interaction)
    if energy < 0.3 or arousal > 0.7:
        decayed *= 0.5

    return max(decayed, 0.0)
