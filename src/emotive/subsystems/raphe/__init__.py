"""Mood subsystem: neurochemical residue of emotional episodes.

Mood is not computed — it's what's LEFT after emotions fade. Each episode
shifts mood dimensions slightly via residue. Between episodes, homeostasis
decays mood back toward temperament baseline.

Subscribes to EPISODE_CREATED on the EventBus. Updates mood state.
Publishes MOOD_UPDATED.

Brain analog: sustained neurotransmitter levels modulated by emotional events.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import EPISODE_CREATED, MOOD_UPDATED
from emotive.subsystems import Subsystem

from .homeostasis import apply_homeostasis
from .residue import MOOD_DIMENSIONS, compute_residue
from .state import load_mood, save_mood

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("raphe")


class MoodSubsystem(Subsystem):
    """Mood as neurochemical residue."""

    name = "mood"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        self._current: dict[str, float] = {dim: 0.5 for dim in MOOD_DIMENSIONS}
        self._enabled = app.config_manager.get().layers.mood
        self._episode_count: int = 0
        mood_config = app.config_manager.get().mood
        self._tick_interval: int = mood_config.homeostasis_tick_interval
        self._tick_hours: float = mood_config.homeostasis_tick_hours

    def _register_handlers(self) -> None:
        """Subscribe to episode creation events."""
        self._bus.subscribe(EPISODE_CREATED, self._on_episode)

    def load(self) -> dict[str, float]:
        """Load mood from DB with homeostasis applied."""
        if not self._enabled:
            return self._current

        session = self._app.session_factory()
        try:
            self._current = load_mood(session)
            return self._current
        finally:
            session.close()

    def save(self) -> None:
        """Persist current mood to DB."""
        if not self._enabled:
            return

        session = self._app.session_factory()
        try:
            save_mood(session, self._current)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to save mood")
        finally:
            session.close()

    def _on_episode(self, event_type: str, data: dict) -> None:
        """Episode created → apply residue to mood state.

        This is the core mechanism: each episode leaves neurochemical
        residue that shifts mood dimensions. The residue accumulates.
        Homeostasis decays it back over time.

        Uses a SINGLE DB session for all operations (save + history +
        homeostasis) to avoid connection pool pressure. Previously each
        step opened its own session, causing pool exhaustion at ~31
        exchanges.
        """
        if not self._enabled:
            return

        emotion = data.get("primary_emotion", "neutral")
        intensity = data.get("intensity", 0.0)

        residue = compute_residue(emotion, intensity)
        if not residue:
            return

        # Apply residue to current mood
        previous = dict(self._current)
        for dim, delta in residue.items():
            if dim in self._current:
                self._current[dim] = max(0.0, min(1.0, self._current[dim] + delta))

        # Within-session homeostasis tick: periodically decay toward baseline
        # so mood doesn't just accumulate residue indefinitely during a session
        self._episode_count += 1
        needs_homeostasis = self._episode_count % self._tick_interval == 0

        # Single session for all DB operations: save + history + homeostasis
        session = self._app.session_factory()
        try:
            # Homeostasis (needs temperament from DB)
            if needs_homeostasis:
                self._apply_within_session_homeostasis_with_session(session)

            # Save mood state
            save_mood(session, self._current)

            # Record history snapshot
            self._record_history_with_session(session, emotion, intensity)

            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to update mood on episode")
        finally:
            session.close()

        # Publish mood update (outside session scope — DB handler will
        # open its own short-lived session)
        self._bus.publish(
            MOOD_UPDATED,
            {
                "previous": previous,
                "current": dict(self._current),
                "residue": residue,
                "source_emotion": emotion,
                "source_intensity": intensity,
            },
        )

        logger.info(
            "Mood updated: %s (%.2f) → %s",
            emotion,
            intensity,
            ", ".join(f"{d}={v:.3f}" for d, v in self._current.items()
                      if abs(v - 0.5) > 0.01),
        )

    def _record_history_with_session(
        self, session, emotion: str, intensity: float
    ) -> None:
        """Record mood snapshot using an existing session."""
        from emotive.db.models.mood import MoodHistory

        snapshot = MoodHistory(
            novelty_seeking=self._current.get("novelty_seeking", 0.5),
            social_bonding=self._current.get("social_bonding", 0.5),
            analytical_depth=self._current.get("analytical_depth", 0.5),
            playfulness=self._current.get("playfulness", 0.5),
            caution=self._current.get("caution", 0.5),
            expressiveness=self._current.get("expressiveness", 0.5),
            source_emotion=emotion,
            source_intensity=intensity,
        )
        session.add(snapshot)

    def _record_history(self, emotion: str, intensity: float) -> None:
        """Record mood snapshot for research tracking (standalone session)."""
        session = self._app.session_factory()
        try:
            self._record_history_with_session(session, emotion, intensity)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to record mood history")
        finally:
            session.close()

    def _apply_within_session_homeostasis_with_session(self, session) -> None:
        """Run homeostasis using an existing session."""
        from emotive.db.models.temperament import Temperament

        temp = session.get(Temperament, 1)
        temperament = {}
        if temp:
            for dim in MOOD_DIMENSIONS:
                temperament[dim] = getattr(temp, dim, 0.5)
        else:
            temperament = {dim: 0.5 for dim in MOOD_DIMENSIONS}

        self._current = apply_homeostasis(
            self._current, temperament, self._tick_hours,
        )
        logger.info(
            "Within-session homeostasis tick (episode %d, %.2fh): %s",
            self._episode_count,
            self._tick_hours,
            ", ".join(f"{d}={v:.3f}" for d, v in self._current.items()
                      if abs(v - 0.5) > 0.01),
        )

    def _apply_within_session_homeostasis(self) -> None:
        """Run a mini homeostasis pass during a session (standalone session).

        Simulates the body normalizing during sustained wakefulness:
        cortisol drops after the threat passes, heart rate slows.
        Uses temperament baseline from DB if available.
        """
        session = self._app.session_factory()
        try:
            self._apply_within_session_homeostasis_with_session(session)
        except Exception:
            logger.exception("Within-session homeostasis failed")
        finally:
            session.close()

    @property
    def current(self) -> dict[str, float]:
        """Current mood state (in-RAM)."""
        return dict(self._current)

    def get_modulated_sensitivity(self, base_sensitivity: float) -> float:
        """Mood modulates amygdala sensitivity.

        Low mood (below baseline) → higher sensitivity → negative events
        feel MORE negative. High mood → lower sensitivity → more resilient.

        Brain analog: depleted serotonin lowers the threshold for
        emotional reactivity.
        """
        if not self._enabled:
            return base_sensitivity

        # Average deviation from baseline (0.5)
        avg_valence = sum(self._current.values()) / len(self._current)
        # Below 0.5 = negative mood → boost sensitivity
        # Above 0.5 = positive mood → reduce sensitivity
        mood_modifier = (0.5 - avg_valence) * 0.3
        return max(0.1, min(0.9, base_sensitivity + mood_modifier))
