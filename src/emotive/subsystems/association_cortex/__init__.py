"""Association Cortex subsystem: automatic memory retrieval.

Wraps the existing recall_memories() pipeline. Takes a pre-computed
embedding from the thalamus (embed once, share the vector).

Brain analog: association cortices — pattern completion from partial cues.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from emotive.logging import get_logger
from emotive.memory.base import recall_memories
from emotive.runtime.event_bus import MEMORIES_RECALLED
from emotive.subsystems import Subsystem

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.runtime.event_bus import EventBus

logger = get_logger("association_cortex")


def _apply_mood_preactivation(
    results: list[dict], mood: dict[str, float]
) -> list[dict]:
    """Pre-activate mood-congruent memories.

    The brain doesn't bias search results with weights. It pre-activates
    memory traces that match the current neurochemical state. Sad mood →
    sad memories were already closer to firing threshold → they surface
    more easily.

    We boost the final_rank of memories whose emotional tags match the
    mood's dominant direction, then re-sort.
    """
    # Determine mood valence: average of all dimensions
    avg_mood = sum(mood.values()) / max(len(mood), 1)
    # Positive mood (>0.5) → boost positive-tagged memories
    # Negative mood (<0.5) → boost negative-tagged memories
    mood_is_positive = avg_mood > 0.52
    mood_is_negative = avg_mood < 0.48

    if not mood_is_positive and not mood_is_negative:
        return results  # Near baseline — no pre-activation

    positive_emotions = {"joy", "trust", "awe", "surprise"}
    negative_emotions = {"sadness", "anger", "fear", "disgust"}

    boost = abs(avg_mood - 0.5) * 0.2  # Small boost, max ~0.1

    for mem in results:
        tags = set(mem.get("tags") or [])
        emotion = mem.get("primary_emotion", "")

        if mood_is_positive and (tags & positive_emotions or emotion in positive_emotions):
            mem["final_rank"] = mem.get("final_rank", 0) + boost
        elif mood_is_negative and (tags & negative_emotions or emotion in negative_emotions):
            mem["final_rank"] = mem.get("final_rank", 0) + boost

    # Re-sort by boosted rank
    results.sort(key=lambda m: m.get("final_rank", 0), reverse=True)
    return results


class AssociationCortex(Subsystem):
    """Memory retrieval subsystem — auto-recall from input."""

    name = "association_cortex"

    def recall(
        self,
        input_embedding: list[float],
        query_text: str,
        conversation_id: uuid.UUID | None = None,
        mood: dict[str, float] | None = None,
    ) -> list[dict]:
        """Retrieve relevant memories using pre-computed embedding.

        Uses the existing recall_memories() pipeline with config-based
        weights. The embedding is shared from the thalamus — no duplicate
        embedding work.
        """
        config = self._app.config_manager.get()

        if not config.auto_recall.enabled:
            return []

        session = self._app.session_factory()
        try:
            results = recall_memories(
                session,
                self._app.embedding_service,
                query=query_text,
                query_embedding=input_embedding,
                limit=config.auto_recall.limit,
                include_spreading=config.auto_recall.include_spreading,
                w_semantic=config.retrieval_weights.semantic,
                w_recency=config.retrieval_weights.recency,
                w_activation=config.retrieval_weights.spreading_activation,
                w_significance=config.retrieval_weights.significance,
                spreading_hops=config.spreading_activation.hops,
                spreading_decay=config.spreading_activation.decay_per_hop,
                event_bus=self._bus,
                conversation_id=conversation_id,
            )
            session.commit()

            # Mood-congruent pre-activation: memories matching current
            # mood valence were already closer to firing — boost them.
            # Brain analog: depleted serotonin pre-activates negative traces.
            if results and mood and config.layers.mood:
                results = _apply_mood_preactivation(results, mood)

            if results:
                self._bus.publish(
                    MEMORIES_RECALLED,
                    {
                        "count": len(results),
                        "top_similarity": results[0].get("similarity", 0)
                        if results else 0,
                    },
                    conversation_id=conversation_id,
                )
                logger.info("Auto-recalled %d memories", len(results))

            return results
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
