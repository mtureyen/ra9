"""Association Cortex subsystem: automatic memory retrieval.

Phase Anamnesis upgrade: three-phase neural retrieval replaces
one-shot cosine similarity. Falls back to legacy recall_memories()
if anamnesis is not enabled in config.

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
    from emotive.subsystems.entorhinal.separation import PatternSeparator
    from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache
    from emotive.subsystems.hippocampus.retrieval.pipeline import RetrievalResult
    from emotive.subsystems.hippocampus.retrieval.state import RetrievalState

logger = get_logger("association_cortex")


def _apply_mood_preactivation(
    results: list[dict], mood: dict[str, float]
) -> list[dict]:
    """Pre-activate mood-congruent memories (legacy path).

    The brain doesn't bias search results with weights. It pre-activates
    memory traces that match the current neurochemical state. Sad mood →
    sad memories were already closer to firing threshold → they surface
    more easily.
    """
    avg_mood = sum(mood.values()) / max(len(mood), 1)
    mood_is_positive = avg_mood > 0.52
    mood_is_negative = avg_mood < 0.48

    if not mood_is_positive and not mood_is_negative:
        return results

    positive_emotions = {"joy", "trust", "awe", "surprise"}
    negative_emotions = {"sadness", "anger", "fear", "disgust"}

    boost = abs(avg_mood - 0.5) * 0.2

    for mem in results:
        tags = set(mem.get("tags") or [])
        emotion = mem.get("primary_emotion", "")

        if mood_is_positive and (tags & positive_emotions or emotion in positive_emotions):
            mem["final_rank"] = mem.get("final_rank", 0) + boost
        elif mood_is_negative and (tags & negative_emotions or emotion in negative_emotions):
            mem["final_rank"] = mem.get("final_rank", 0) + boost

    results.sort(key=lambda m: m.get("final_rank", 0), reverse=True)
    return results


class AssociationCortex(Subsystem):
    """Memory retrieval subsystem — auto-recall from input.

    Supports two modes:
    - Legacy: one-shot cosine similarity (recall_memories)
    - Anamnesis: three-phase neural pipeline (run_retrieval)
    """

    name = "association_cortex"

    def recall(
        self,
        input_embedding: list[float],
        query_text: str,
        conversation_id: uuid.UUID | None = None,
        mood: dict[str, float] | None = None,
    ) -> list[dict]:
        """Legacy retrieval path — one-shot cosine similarity.

        Used when config.layers.anamnesis is False.
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
                logger.info("Auto-recalled %d memories (legacy)", len(results))

            return results
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def recall_anamnesis(
        self,
        input_embedding: list[float],
        query_text: str,
        retrieval_state: RetrievalState,
        person_cache: PersonNodeCache,
        separator: PatternSeparator,
        *,
        conversation_id: uuid.UUID | None = None,
        mood: dict[str, float] | None = None,
        prediction_error: float = 0.0,
        emotional_intensity: float = 0.0,
        person_trust: float = 0.5,
        comfort: float = 0.5,
    ) -> RetrievalResult:
        """Phase Anamnesis retrieval — three-phase neural pipeline.

        Used when config.layers.anamnesis is True.
        Returns full RetrievalResult with conscious/unconscious split,
        familiarity/recollection signals, TOT state, effort.
        """
        from emotive.subsystems.hippocampus.retrieval.pipeline import run_retrieval

        config = self._app.config_manager.get()

        if not config.auto_recall.enabled:
            from emotive.subsystems.hippocampus.retrieval.pipeline import RetrievalResult
            return RetrievalResult()

        session = self._app.session_factory()
        try:
            result = run_retrieval(
                db_session=session,
                query_text=query_text,
                query_embedding=input_embedding,
                retrieval_state=retrieval_state,
                person_cache=person_cache,
                separator=separator,
                mood=mood,
                prediction_error=prediction_error,
                emotional_intensity=emotional_intensity,
                conscious_limit=config.workspace.max_context_memories
                if hasattr(config, "workspace")
                else 5,
                person_trust=person_trust,
                comfort=comfort,
                embedding_service=self._app.embedding_service,
                conversation_id=conversation_id,
            )
            session.commit()

            if result.conscious:
                self._bus.publish(
                    MEMORIES_RECALLED,
                    {
                        "count": len(result.conscious),
                        "top_similarity": result.conscious[0].get("similarity", 0)
                        if result.conscious else 0,
                        "strategy": result.strategy,
                        "effort": result.effort,
                        "tot": result.tot_active,
                        "familiarity": result.familiarity_score,
                        "recollection": result.recollection_score,
                    },
                    conversation_id=conversation_id,
                )
                logger.info(
                    "Anamnesis recalled %d memories (strategy=%s, effort=%.2f)",
                    len(result.conscious),
                    result.strategy,
                    result.effort,
                )

            return result
        except Exception:
            session.rollback()
            logger.exception("Anamnesis retrieval failed, falling back to legacy")
            # Fallback: return empty result, thalamus can try legacy
            from emotive.subsystems.hippocampus.retrieval.pipeline import RetrievalResult
            return RetrievalResult()
        finally:
            session.close()
