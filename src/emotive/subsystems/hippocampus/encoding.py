"""Unconscious encoding: automatic memory storage based on appraisal intensity.

When appraisal intensity exceeds the threshold, the hippocampus silently
encodes the exchange as an episodic memory. The LLM never sees this happen.
Rate-limited by max_per_exchange and cooldown_seconds.

Brain analog: hippocampal encoding, arousal-gated by amygdala output.
"""

from __future__ import annotations

import time
import uuid

from sqlalchemy.orm import Session

from emotive.config.schema import UnconsciousEncodingConfig
from emotive.db.models.memory import Memory
from emotive.embeddings.service import EmbeddingService
from emotive.layers.appraisal import AppraisalResult, run_appraisal
from emotive.layers.episodes import create_episode
from emotive.logging import get_logger
from emotive.memory.episodic import store_episodic_from_episode
from emotive.runtime.event_bus import EventBus

logger = get_logger("hippocampus.encoding")


class UnconsciousEncoder:
    """Manages rate-limited unconscious encoding."""

    def __init__(self, config: UnconsciousEncodingConfig) -> None:
        self._threshold = config.intensity_threshold
        self._max_per_exchange = config.max_per_exchange
        self._cooldown = config.cooldown_seconds
        self._last_encode_time: float = 0.0
        self._exchange_count: int = 0

    def reset_exchange(self) -> None:
        """Call at the start of each exchange to reset per-exchange counter."""
        self._exchange_count = 0

    def should_encode(self, intensity: float) -> bool:
        """Check if encoding should proceed based on thresholds and limits."""
        if intensity < self._threshold:
            return False
        if self._exchange_count >= self._max_per_exchange:
            return False
        elapsed = time.monotonic() - self._last_encode_time
        if elapsed < self._cooldown and self._last_encode_time > 0:
            return False
        return True

    # Tags to exclude from context inheritance
    _EXCLUDED_TAGS = frozenset({"gist", "conversation_summary", "conscious_intent"})

    def encode(
        self,
        session: Session,
        embedding_service: EmbeddingService,
        appraisal: AppraisalResult,
        content: str,
        *,
        conversation_id: uuid.UUID | None = None,
        sensitivity: float = 0.5,
        resilience: float = 0.5,
        context_tags: list[str] | None = None,
        event_bus: EventBus | None = None,
    ) -> tuple[Memory | None, uuid.UUID | None]:
        """Encode an exchange as episode + episodic memory if significant.

        context_tags: tags from co-active memories (what the brain was
        thinking about during this exchange). Inherited by the new memory.

        Returns (memory, episode_id) or (None, None) if below threshold.
        """
        if not self.should_encode(appraisal.intensity):
            return None, None

        # Build tags: emotion + context from co-active memories
        tags = [appraisal.primary_emotion]
        if context_tags:
            for tag in context_tags:
                if tag not in tags and tag not in self._EXCLUDED_TAGS:
                    tags.append(tag)
                if len(tags) >= 6:
                    break

        # ACC analog: check for conflict with established identity memories
        from .conflict import detect_conflict

        conflict_score = detect_conflict(session, embedding_service, content[:500])
        conflict_detected = conflict_score > 0.6

        if conflict_detected:
            tags.append("contradiction")
            logger.info(
                "ACC conflict detected (%.2f) — encoding weakened for: %s",
                conflict_score,
                content[:80],
            )

        # Create episode
        episode = create_episode(
            session,
            appraisal,
            trigger_event=content[:500],
            trigger_source="user_message",
            conversation_id=conversation_id,
            event_bus=event_bus,
        )

        # Encode as episodic memory
        memory = store_episodic_from_episode(
            session,
            embedding_service,
            episode=episode,
            content=content[:500],
            conversation_id=conversation_id,
            tags=tags,
            event_bus=event_bus,
        )

        # ACC weakening: reduce protection so contradictory memories fade faster
        if conflict_detected:
            memory.decay_protection = 0.0  # no protection — fades fast
            memory.confidence = 0.3  # low confidence
            if memory.metadata_ is None:
                memory.metadata_ = {}
            memory.metadata_["conflict_score"] = conflict_score
            session.flush()

        self._exchange_count += 1
        self._last_encode_time = time.monotonic()

        logger.info(
            "Unconscious encoding: %s (%.2f) → episode %s, memory %s",
            appraisal.primary_emotion,
            appraisal.intensity,
            episode.id,
            memory.id,
        )

        return memory, episode.id
