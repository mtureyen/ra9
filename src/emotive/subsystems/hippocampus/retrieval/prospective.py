"""Prospective memory — E21.

Remembering to remember. Dormant intention-cue pairs that fire
automatically when the trigger cue appears.

Two pathways:
  1. Top-down monitoring (costly, PFC-driven)
  2. Bottom-up spontaneous (cue triggers retrieval automatically)

We implement pathway 2: store trigger-cue pairs, check at retrieval.

Brain analog: hippocampal-PFC intention storage + cue-driven retrieval.
Sources: McDaniel & Einstein, Dual Pathways; PMC4500919.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.logging import get_logger

logger = get_logger("hippocampus.retrieval.prospective")


def load_prospective_memories(db_session: Session) -> list[dict]:
    """Load all pending prospective memories at session boot.

    Prospective memories are stored as memory_type='prospective'
    with trigger_cue and intended_action in metadata.
    """
    stmt = (
        select(Memory)
        .where(Memory.memory_type == "prospective")
        .where(Memory.is_archived.is_(False))
    )
    rows = db_session.execute(stmt).scalars().all()

    active = []
    now = datetime.now(timezone.utc)
    for m in rows:
        meta = m.metadata_ or {}
        expires = meta.get("expires_at")
        if expires:
            try:
                exp_dt = datetime.fromisoformat(expires)
                if exp_dt < now:
                    # Expired intention — archive it
                    m.is_archived = True
                    continue
            except (ValueError, TypeError):
                pass

        active.append({
            "id": m.id,
            "trigger_cue": meta.get("trigger_cue", ""),
            "intended_action": meta.get("intended_action", ""),
            "content": m.content,
        })

    db_session.flush()
    return active


def check_prospective_triggers(
    prospective_memories: list[dict],
    current_input: str,
    detected_person: str | None = None,
) -> list[str]:
    """Check if any prospective memories should trigger.

    Returns list of intention descriptions to inject as inner voice nudges.
    """
    triggered = []
    input_lower = current_input.lower()

    for pm in prospective_memories:
        cue = pm.get("trigger_cue", "").lower()
        if not cue:
            continue

        # Check if trigger cue appears in input or matches person
        if cue in input_lower or cue == (detected_person or ""):
            action = pm.get("intended_action", pm.get("content", ""))
            triggered.append(f"You wanted to: {action}")
            logger.info("Prospective memory triggered: %s → %s", cue, action)

    return triggered


def store_prospective_memory(
    db_session: Session,
    embedding_service,
    trigger_cue: str,
    intended_action: str,
    conversation_id: uuid.UUID | None = None,
    expiry_days: int = 7,
) -> Memory:
    """Store a new prospective memory (intention).

    Created when inner speech identifies a future intention:
    "I should remember to ask Lena about X."
    """
    from emotive.memory.base import store_memory

    now = datetime.now(timezone.utc)
    expires = (now + timedelta(days=expiry_days)).isoformat()

    return store_memory(
        db_session,
        embedding_service,
        content=f"Intention: {intended_action} (trigger: {trigger_cue})",
        memory_type="prospective",
        conversation_id=conversation_id,
        tags=["prospective", trigger_cue.lower()],
        metadata={
            "trigger_cue": trigger_cue.lower(),
            "intended_action": intended_action,
            "expires_at": expires,
        },
        decay_rate=0.001,  # intentions fade relatively fast
    )
