"""Retrieval state — persistent across exchanges within a session.

Tracks everything the retrieval system needs to remember between
exchanges: context vector, what was recalled, what's suppressed,
what strategy is active, tip-of-the-tongue states.

Brain analog: hippocampal-prefrontal state that persists across
retrieval attempts within a conversation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .context_vector import ContextVector


@dataclass
class TOTState:
    """Tip-of-the-tongue: partial retrieval without full access."""

    active: bool = False
    partial_person: str | None = None  # might be wrong
    partial_emotion: str | None = None
    partial_time: str | None = None  # "recent" / "a while ago"
    near_miss_ids: list[uuid.UUID] = field(default_factory=list)
    confidence: float = 0.0  # how close we got


@dataclass
class RIFEntry:
    """Retrieval-induced forgetting: suppressed competitor."""

    memory_id: uuid.UUID
    suppression: float  # 0.0 - 1.0
    decay_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def current_suppression(self, now: datetime | None = None) -> float:
        """Suppression decays exponentially with τ = 6 hours."""
        if now is None:
            now = datetime.now(timezone.utc)
        elapsed_hours = (now - self.decay_start).total_seconds() / 3600
        tau = 6.0
        import math

        return self.suppression * math.exp(-elapsed_hours / tau)


@dataclass
class RetrievalState:
    """All retrieval state that persists across exchanges.

    Created at session boot. Updated each exchange. Cleared at session end.
    """

    # Context vector (TCM model)
    context: ContextVector = field(default_factory=ContextVector)

    # What was recalled last exchange (for cascade seeding)
    previously_recalled: list[uuid.UUID] = field(default_factory=list)

    # Memories 6-20 in ranking — available for cascade next exchange
    unconscious_pool: list[uuid.UUID] = field(default_factory=list)

    # Who/what we're currently discussing
    topic_person: str | None = None

    # Current retrieval strategy (PERSON/TOPIC/EMOTION/TEMPORAL/BROAD)
    strategy: str = "BROAD"

    # RIF: suppressed competitors + decay tracking
    rif_suppressed: dict[uuid.UUID, RIFEntry] = field(default_factory=dict)

    # How many exchanges we've been on this topic
    exchange_count_on_topic: int = 0

    # Tip-of-the-tongue state
    tot: TOTState = field(default_factory=TOTState)

    # Retrieval effort from last exchange (0.0 - 1.0)
    last_effort: float = 0.0

    # E19: Meta-memory encoding cap (1 per session)
    _meta_memory_stored: bool = False

    # Encoding-retrieval mode balance
    encoding_strength: float = 0.5
    retrieval_strength: float = 0.5

    # Familiarity signal from last exchange
    last_familiarity_score: float = 0.0
    last_recollection_score: float = 0.0

    def update_topic(self, person: str | None, topic_changed: bool) -> None:
        """Update topic tracking for the current exchange."""
        if topic_changed or person != self.topic_person:
            self.topic_person = person
            self.exchange_count_on_topic = 1
        else:
            self.exchange_count_on_topic += 1

    def get_theta_iterations(self, base: int = 2, max_iter: int = 5) -> int:
        """Compute retrieval depth from sustained attention.

        More exchanges on same topic → deeper retrieval cascade.
        Modulated by retrieval_strength (encoding-retrieval antagonism).
        """
        depth = base + self.exchange_count_on_topic
        depth = min(depth, max_iter)

        # Scale by retrieval mode strength
        scaled = int(depth * (self.retrieval_strength / 0.5))
        return max(1, min(scaled, max_iter))

    def compute_mode_balance(
        self,
        prediction_error: float,
        emotional_intensity: float,
        is_recall_query: bool,
    ) -> None:
        """Compute encoding-retrieval antagonism balance.

        High novelty/emotion → encoding mode (suppress retrieval).
        Explicit recall query → retrieval mode (raise encoding threshold).

        Brain analog: theta-phase switching in hippocampus.
        Sources: Hasselmo et al. (2013), Siegle & Wilson (2014).
        """
        # Encoding drive: novelty + emotion
        encoding_drive = prediction_error * 0.4 + emotional_intensity * 0.4
        encoding_drive = max(encoding_drive, 0.2)  # minimum encoding

        # Retrieval drive: recall queries boost, default moderate
        retrieval_drive = 0.6 if is_recall_query else 0.3

        # Normalize to sum ~1.0
        total = encoding_drive + retrieval_drive
        self.encoding_strength = encoding_drive / total
        self.retrieval_strength = retrieval_drive / total

    def get_active_rif(self) -> dict[uuid.UUID, float]:
        """Get currently active RIF suppressions (decay applied)."""
        now = datetime.now(timezone.utc)
        active = {}
        expired = []
        for mid, entry in self.rif_suppressed.items():
            current = entry.current_suppression(now)
            if current > 0.01:
                active[mid] = current
            else:
                expired.append(mid)
        # Clean up expired
        for mid in expired:
            del self.rif_suppressed[mid]
        return active

    def add_rif(self, memory_id: uuid.UUID, suppression: float = 0.3) -> None:
        """Suppress a competitor memory via RIF."""
        self.rif_suppressed[memory_id] = RIFEntry(
            memory_id=memory_id,
            suppression=suppression,
        )

    def reset(self) -> None:
        """Reset all state (new session)."""
        self.context.reset()
        self.previously_recalled.clear()
        self.unconscious_pool.clear()
        self.topic_person = None
        self.strategy = "BROAD"
        self.rif_suppressed.clear()
        self.exchange_count_on_topic = 0
        self.tot = TOTState()
        self.last_effort = 0.0
        self.encoding_strength = 0.5
        self.retrieval_strength = 0.5
        self.last_familiarity_score = 0.0
        self.last_recollection_score = 0.0
        self._meta_memory_stored = False
