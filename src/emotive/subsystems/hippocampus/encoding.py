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
    """Manages rate-limited unconscious encoding with dynamic threshold.

    The encoding threshold is modulated by arousal context — like the
    locus coeruleus releasing norepinephrine. High social significance,
    high novelty, or active emotional episodes lower the threshold,
    making more exchanges significant enough to encode.

    Brain analog: locus coeruleus arousal modulation of hippocampal encoding.
    """

    def __init__(self, config: UnconsciousEncodingConfig) -> None:
        self._base_threshold = config.intensity_threshold
        self._max_per_exchange = config.max_per_exchange
        self._cooldown = config.cooldown_seconds
        self._last_encode_time: float = 0.0
        self._exchange_count: int = 0
        self._episode_heat: float = 0.0  # decays over exchanges

    def reset_exchange(self) -> None:
        """Call at the start of each exchange to reset per-exchange counter."""
        self._exchange_count = 0
        # Episode heat decays each exchange (lingering arousal fades)
        self._episode_heat *= 0.7

    def compute_dynamic_threshold(self, appraisal: AppraisalResult) -> float:
        """Compute the effective encoding threshold based on arousal context.

        Locus coeruleus modulation:
        - High novelty → threshold drops (unexpected content encodes easier)
        - High social significance → threshold drops (personal content encodes easier)
        - Active episode heat → threshold drops (emotional context lingers)

        Returns the effective threshold (lower = more encodes).
        """
        arousal_modifier = (
            appraisal.vector.novelty * 0.15
            + appraisal.vector.social_significance * 0.15
            + self._episode_heat * 0.10
        )
        effective = self._base_threshold - arousal_modifier
        # Floor at 0.15 — even in maximum arousal, need minimal intensity
        return max(effective, 0.15)

    def should_encode(
        self, intensity: float, appraisal: AppraisalResult | None = None
    ) -> bool:
        """Check if encoding should proceed.

        Uses dynamic threshold if appraisal is provided, falls back
        to base threshold otherwise (backwards compatible).
        """
        if appraisal is not None:
            threshold = self.compute_dynamic_threshold(appraisal)
        else:
            threshold = self._base_threshold

        if intensity < threshold:
            return False
        if self._exchange_count >= self._max_per_exchange:
            return False
        elapsed = time.monotonic() - self._last_encode_time
        if elapsed < self._cooldown and self._last_encode_time > 0:
            return False
        return True

    def record_encoding(self, intensity: float) -> None:
        """Record that encoding happened — boosts episode heat."""
        self._exchange_count += 1
        self._last_encode_time = time.monotonic()
        # Successful encoding boosts episode heat (brain stays "hot")
        self._episode_heat = min(self._episode_heat + intensity * 0.5, 1.0)

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
        if not self.should_encode(appraisal.intensity, appraisal):
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

        self.record_encoding(appraisal.intensity)

        logger.info(
            "Unconscious encoding: %s (%.2f, threshold=%.2f) → episode %s, memory %s",
            appraisal.primary_emotion,
            appraisal.intensity,
            self.compute_dynamic_threshold(appraisal),
            episode.id,
            memory.id,
        )

        return memory, episode.id
