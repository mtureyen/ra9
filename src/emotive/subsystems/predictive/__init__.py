"""Predictive processing subsystem: expectations and prediction error.

Generates expectations for the next message and computes prediction error
when the actual message arrives. Expectation generated AFTER each exchange
(for next turn). Error computed at START of next exchange from stored
expectation. First message of session: neutral error (0.5).

No DB needed -- expectations are ephemeral, in RAM only.

Brain analog: predictive coding in the cortex. The brain constantly
generates top-down predictions about incoming sensory data. Surprise
(prediction error) drives learning and attention allocation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.runtime.event_bus import EXPECTATION_STORED, PREDICTION_ERROR_COMPUTED
from emotive.subsystems import Subsystem

from .expectations import compute_prediction_error, generate_expectation_embedding

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("predictive")


class PredictiveProcessor(Subsystem):
    """Generates expectations and computes prediction error.

    Expectation generated AFTER each exchange (for next turn).
    Error computed at START of next exchange from stored expectation.
    First message of session: neutral error (0.5).
    """

    name = "predictive"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        self._expected_embedding: list[float] | None = None
        self._recent_embeddings: list[list[float]] = []  # last 3

    def compute_error(self, input_embedding: list[float]) -> float:
        """Compute prediction error for current input."""
        error = compute_prediction_error(self._expected_embedding, input_embedding)
        self._bus.publish(PREDICTION_ERROR_COMPUTED, {"error": error})
        return error

    def store_expectation(self, response_embedding: list[float]) -> None:
        """Generate and store expectation for next turn. Called in post-processing."""
        self._recent_embeddings.append(response_embedding)
        if len(self._recent_embeddings) > 3:
            self._recent_embeddings = self._recent_embeddings[-3:]
        self._expected_embedding = generate_expectation_embedding(
            response_embedding, self._recent_embeddings
        )
        self._bus.publish(EXPECTATION_STORED, {})

    def reset(self) -> None:
        """Reset for new session."""
        self._expected_embedding = None
        self._recent_embeddings = []
