"""Embodied state subsystem: energy, cognitive load, comfort.

Tracks the body-analog state of the system. Energy depletes with each
exchange (nonlinearly — faster when tired). Cognitive load rises with
complexity. Comfort reflects relational ease.

Subscribes to EPISODE_CREATED. Publishes EMBODIED_STATE_UPDATED.

Brain analog: interoceptive cortex — fatigue, mental effort, social ease.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    EMBODIED_STATE_UPDATED,
    EPISODE_CREATED,
)
from emotive.subsystems import Subsystem

from .dynamics import (
    boost_energy,
    deplete_energy,
    recover_energy,
    update_cognitive_load,
    update_comfort,
)

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("insula")


class EmbodiedSubsystem(Subsystem):
    """Body-analog state tracking."""

    name = "embodied"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        self._energy: float = 1.0
        self._cognitive_load: float = 0.0
        self._comfort: float = 0.5
        super().__init__(app, event_bus)

        cfg = app.config_manager.get().embodied
        self._base_rate: float = cfg.energy_depletion_base
        self._joy_boost: float = cfg.joy_boost
        self._comfort_decay: float = cfg.comfort_decay_rate

    def _register_handlers(self) -> None:
        self._bus.subscribe(EPISODE_CREATED, self._on_episode)

    def load(self) -> dict[str, float]:
        """Load embodied state from DB, apply recovery for elapsed time."""
        from emotive.db.models.embodied_state import EmbodiedState

        session = self._app.session_factory()
        try:
            row = session.get(EmbodiedState, 1)
            if not row:
                return self.to_dict()

            self._energy = row.energy
            self._cognitive_load = row.cognitive_load
            self._comfort = row.comfort

            # Apply between-session energy recovery
            now = datetime.now(timezone.utc)
            updated = row.updated_at
            if updated and updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            if updated:
                hours_elapsed = (now - updated).total_seconds() / 3600.0
                if hours_elapsed > 0:
                    self._energy = recover_energy(self._energy, hours_elapsed)

            return self.to_dict()
        finally:
            session.close()

    def save(self) -> None:
        """Persist current embodied state to DB."""
        from emotive.db.models.embodied_state import EmbodiedState

        session = self._app.session_factory()
        try:
            row = session.get(EmbodiedState, 1)
            if not row:
                row = EmbodiedState(id=1)
                session.add(row)

            row.energy = self._energy
            row.cognitive_load = self._cognitive_load
            row.comfort = self._comfort
            row.updated_at = datetime.now(timezone.utc)
            session.commit()
        except Exception:
            session.rollback()
            logger.exception("Failed to save embodied state")
        finally:
            session.close()

    def update(
        self,
        emotion: str,
        intensity: float,
        cognitive_complexity: float = 0.0,
        num_recalled: int = 0,
    ) -> dict[str, float]:
        """Update embodied state for an exchange.

        Args:
            emotion: Primary emotion from appraisal.
            intensity: Emotion intensity [0, 1].
            cognitive_complexity: Prediction error / complexity [0, 1].
            num_recalled: Number of memories recalled this exchange.

        Returns:
            Updated state dict.
        """
        previous = self.to_dict()

        # Energy: deplete then maybe boost
        self._energy = deplete_energy(self._energy, self._base_rate)
        self._energy = boost_energy(
            self._energy, emotion, intensity, self._joy_boost
        )

        # Cognitive load
        self._cognitive_load = update_cognitive_load(
            self._cognitive_load, num_recalled, cognitive_complexity
        )

        # Comfort
        self._comfort = update_comfort(
            self._comfort, emotion, intensity, self._comfort_decay
        )

        # Save to DB
        self.save()

        # Publish event
        current = self.to_dict()
        self._bus.publish(
            EMBODIED_STATE_UPDATED,
            {
                "previous": previous,
                "current": current,
                "source_emotion": emotion,
                "source_intensity": intensity,
            },
        )

        logger.info(
            "Embodied state: energy=%.2f load=%.2f comfort=%.2f",
            self._energy,
            self._cognitive_load,
            self._comfort,
        )

        return current

    def _on_episode(self, event_type: str, data: dict) -> None:
        """Episode created -> update embodied state."""
        emotion = data.get("primary_emotion", "neutral")
        intensity = data.get("intensity", 0.0)
        prediction_error = data.get("prediction_error", 0.0)
        num_recalled = data.get("num_recalled", 0)

        self.update(emotion, intensity, prediction_error, num_recalled)

    def to_dict(self) -> dict[str, float]:
        """Return current state as dict."""
        return {
            "energy": self._energy,
            "cognitive_load": self._cognitive_load,
            "comfort": self._comfort,
        }

    @property
    def energy(self) -> float:
        return self._energy

    @property
    def cognitive_load(self) -> float:
        return self._cognitive_load

    @property
    def comfort(self) -> float:
        return self._comfort
