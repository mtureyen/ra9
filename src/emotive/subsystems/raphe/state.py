"""Mood state: load, save, and convert between DB model and dict.

The mood state is a singleton row in the database (like temperament).
It persists across sessions. Homeostasis is applied on load based on
elapsed time since last update.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.orm import Session

from emotive.db.models.mood import MoodState
from emotive.db.models.temperament import Temperament

from .homeostasis import apply_homeostasis
from .residue import MOOD_DIMENSIONS


def load_mood(session: Session) -> dict[str, float]:
    """Load current mood from DB, apply homeostasis for elapsed time."""
    mood_row = session.get(MoodState, 1)
    if not mood_row:
        return {dim: 0.5 for dim in MOOD_DIMENSIONS}

    # Get temperament baseline for homeostasis
    temp = session.get(Temperament, 1)
    temperament = {}
    if temp:
        for dim in MOOD_DIMENSIONS:
            temperament[dim] = getattr(temp, dim, 0.5)
    else:
        temperament = {dim: 0.5 for dim in MOOD_DIMENSIONS}

    # Current mood values
    mood = {}
    for dim in MOOD_DIMENSIONS:
        mood[dim] = getattr(mood_row, dim, 0.5)

    # Apply homeostasis for time elapsed since last update
    now = datetime.now(timezone.utc)
    updated = mood_row.updated_at
    if updated and updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    if updated:
        hours_elapsed = (now - updated).total_seconds() / 3600.0
        if hours_elapsed > 0:
            mood = apply_homeostasis(mood, temperament, hours_elapsed)

    return mood


def save_mood(session: Session, mood: dict[str, float]) -> None:
    """Save current mood to DB."""
    mood_row = session.get(MoodState, 1)
    if not mood_row:
        mood_row = MoodState(id=1)
        session.add(mood_row)

    for dim in MOOD_DIMENSIONS:
        val = max(0.0, min(1.0, mood.get(dim, 0.5)))
        setattr(mood_row, dim, val)

    mood_row.updated_at = datetime.now(timezone.utc)
    session.flush()


def mood_to_dict(mood_row: MoodState) -> dict[str, float]:
    """Convert MoodState model to dict."""
    return {dim: getattr(mood_row, dim, 0.5) for dim in MOOD_DIMENSIONS}
