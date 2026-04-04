"""ACC repetition monitor: detects stuck response patterns.

The anterior cingulate cortex monitors INFORMATION GAIN, not topic
repetition. Same topic with new info = fine. Same response with zero
new info = stuck. When stuck is detected, the locus coeruleus triggers
exploration via a novelty nudge on the next exchange.

Brain analog: ACC repetition monitoring (function 2, alongside conflict
detection). Locus coeruleus norepinephrine release for exploration shift.
"""

from __future__ import annotations

from collections import deque

from emotive.logging import get_logger
from emotive.subsystems.amygdala.fast_pass import cosine_similarity

logger = get_logger("acc.repetition")

# Response similarity above this = generating the same output
RESPONSE_SIMILARITY_THRESHOLD = 0.85

# User input novelty above this = they broke the loop themselves
USER_NOVELTY_THRESHOLD = 0.5


class RepetitionMonitor:
    """ACC repetition detection. Tracks response similarity and novelty scores.

    Two signals for "stuck":
    1. High response similarity (>0.85) between last 2-3 responses
    2. Declining novelty (3 consecutive drops in amygdala novelty score)

    NOT stuck: same topic but with rising novelty (deep conversation).
    """

    def __init__(self, max_history: int = 3) -> None:
        self._response_embeddings: deque[list[float]] = deque(maxlen=max_history)
        self._novelty_scores: deque[float] = deque(maxlen=max_history)
        self._stuck: bool = False

    def update(self, response_embedding: list[float], novelty: float) -> bool:
        """Called after each exchange. Returns True if stuck detected.

        response_embedding: embedding of the LLM's response
        novelty: the novelty score from the amygdala's appraisal
        """
        self._response_embeddings.append(response_embedding)
        self._novelty_scores.append(novelty)
        self._stuck = self._check_stuck()

        if self._stuck:
            logger.info(
                "ACC repetition detected — loop flagged for novelty nudge"
            )

        return self._stuck

    def _check_stuck(self) -> bool:
        """Check both signals for stuckness."""
        if len(self._response_embeddings) < 2:
            return False

        # Signal 1: High response similarity (model generating same output)
        latest = self._response_embeddings[-1]
        for prev in list(self._response_embeddings)[:-1]:
            sim = cosine_similarity(latest, prev)
            if sim > RESPONSE_SIMILARITY_THRESHOLD:
                logger.info("Response similarity %.2f > threshold", sim)
                return True

        # Signal 2: Declining novelty (3 consecutive drops = losing info gain)
        if len(self._novelty_scores) >= 3:
            scores = list(self._novelty_scores)
            if scores[-1] < scores[-2] < scores[-3]:
                logger.info(
                    "Declining novelty: %.2f → %.2f → %.2f",
                    scores[-3], scores[-2], scores[-1],
                )
                return True

        return False

    def cancel_nudge(self, input_novelty: float) -> bool:
        """Check if user's new input breaks the loop.

        If the user introduced something new (high novelty), they broke
        the loop themselves — cancel the nudge. Social feedback overrides.
        """
        if input_novelty > USER_NOVELTY_THRESHOLD:
            self._stuck = False
            logger.info(
                "User broke loop (novelty=%.2f) — nudge cancelled", input_novelty
            )
            return True
        return False

    @property
    def is_stuck(self) -> bool:
        """Whether a loop was detected on the previous exchange."""
        return self._stuck

    def reset(self) -> None:
        """Reset on new session."""
        self._response_embeddings.clear()
        self._novelty_scores.clear()
        self._stuck = False
