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


class AssociationCortex(Subsystem):
    """Memory retrieval subsystem — auto-recall from input."""

    name = "association_cortex"

    def recall(
        self,
        input_embedding: list[float],
        query_text: str,
        conversation_id: uuid.UUID | None = None,
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
