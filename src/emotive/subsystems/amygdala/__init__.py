"""Amygdala subsystem: two-pass emotional appraisal.

Fast pass (pre-LLM): embedding similarity against emotion prototypes.
Slow pass (post-LLM): full exchange against situation prototypes, with reappraisal.

Brain analog: amygdala — relevance detector, not just threat detector.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from emotive.layers.appraisal import AppraisalResult
from emotive.logging import get_logger
from emotive.runtime.event_bus import (
    APPRAISAL_COMPLETE,
    FAST_APPRAISAL_COMPLETE,
    SOCIAL_PERCEPTION_COMPLETE,
)
from emotive.subsystems import Subsystem

from .fast_pass import run_fast_pass
from .prototypes import (
    FAST_PROTOTYPE_TEXTS,
    SLOW_PROTOTYPE_TEXTS,
    compute_prototype_embeddings,
)
from .slow_pass import run_slow_pass
from .social_perception import (
    USER_STATE_PROTOTYPE_TEXTS,
    compute_social_perception_prototypes,
    run_social_perception,
)

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("amygdala")


class Amygdala(Subsystem):
    """Emotional appraisal subsystem — two-pass architecture."""

    name = "amygdala"

    def __init__(self, app: AppContext, event_bus: EventBus) -> None:
        super().__init__(app, event_bus)
        self._fast_prototypes: dict[str, list[float]] = {}
        self._slow_prototypes: dict[str, list[float]] = {}
        self._social_prototypes: dict[str, list[float]] = {}
        self._initialized = False

    def _ensure_prototypes(self) -> None:
        """Lazy-initialize prototype embeddings on first use."""
        if self._initialized:
            return
        logger.info("Computing emotion prototype embeddings...")
        es = self._app.embedding_service
        self._fast_prototypes = compute_prototype_embeddings(
            FAST_PROTOTYPE_TEXTS, es
        )
        self._slow_prototypes = compute_prototype_embeddings(
            SLOW_PROTOTYPE_TEXTS, es
        )
        self._social_prototypes = compute_social_perception_prototypes(
            USER_STATE_PROTOTYPE_TEXTS, es
        )
        self._initialized = True
        logger.info(
            "Prototypes ready: %d fast, %d slow, %d social",
            len(self._fast_prototypes),
            len(self._slow_prototypes),
            len(self._social_prototypes),
        )

    def fast_pass(
        self,
        input_embedding: list[float],
        *,
        sensitivity: float = 0.5,
        resilience: float = 0.5,
        formative_threshold: float = 0.8,
    ) -> AppraisalResult:
        """Pre-LLM appraisal. Takes pre-computed input embedding.

        ~5ms. Cosine similarity against emotion prototypes.
        """
        self._ensure_prototypes()
        result = run_fast_pass(
            input_embedding,
            self._fast_prototypes,
            sensitivity=sensitivity,
            resilience=resilience,
            formative_threshold=formative_threshold,
        )

        # Social perception: read the user's emotional state
        user_state, user_state_confidence = run_social_perception(
            input_embedding, self._social_prototypes
        )
        result.user_state = user_state
        result.user_state_confidence = user_state_confidence

        self._bus.publish(
            FAST_APPRAISAL_COMPLETE,
            {
                "primary_emotion": result.primary_emotion,
                "intensity": result.intensity,
            },
        )
        self._bus.publish(
            SOCIAL_PERCEPTION_COMPLETE,
            {
                "user_state": user_state,
                "user_state_confidence": user_state_confidence,
            },
        )
        return result

    def slow_pass(
        self,
        user_message: str,
        llm_response: str,
        fast_result: AppraisalResult,
        *,
        sensitivity: float = 0.5,
        resilience: float = 0.5,
        formative_threshold: float = 0.8,
    ) -> AppraisalResult:
        """Post-LLM appraisal. Embeds full exchange, compares against
        situation prototypes. Can override fast pass (reappraisal).

        ~15ms. The slow pass embeds the exchange itself.
        """
        self._ensure_prototypes()
        result = run_slow_pass(
            user_message,
            llm_response,
            self._slow_prototypes,
            fast_result,
            self._app.embedding_service,
            sensitivity=sensitivity,
            resilience=resilience,
            formative_threshold=formative_threshold,
        )

        reappraised = result is not fast_result
        self._bus.publish(
            APPRAISAL_COMPLETE,
            {
                "primary_emotion": result.primary_emotion,
                "intensity": result.intensity,
                "reappraised": reappraised,
            },
        )
        if reappraised:
            logger.info(
                "Reappraisal: %s (%.2f) → %s (%.2f)",
                fast_result.primary_emotion,
                fast_result.intensity,
                result.primary_emotion,
                result.intensity,
            )
        return result
