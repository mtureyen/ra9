"""Global Workspace subsystem: attention bottleneck.

Collects signals from all subsystems (amygdala, memory, conflict,
prediction, mood), ranks them by salience, and broadcasts the top-N
into LLM context. Sub-threshold signals remain unconscious (logged only).

Brain analog: Global Workspace Theory (Baars) -- conscious access
requires winning the competition for workspace broadcast.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from emotive.logging import get_logger
from emotive.runtime.event_bus import WORKSPACE_BROADCAST
from emotive.subsystems import Subsystem

from .salience import compute_salience, rank_and_select
from .signals import WorkspaceOutput, WorkspaceSignal

if TYPE_CHECKING:
    from emotive.app_context import AppContext
    from emotive.config.schema import WorkspaceConfig
    from emotive.runtime.event_bus import EventBus

logger = get_logger("workspace")


class GlobalWorkspace(Subsystem):
    """Attention bottleneck -- selects what enters conscious processing."""

    name = "workspace"

    def broadcast(
        self,
        recalled_memories: list[dict],
        appraisal: Any,
        prediction_error: float,
        mood: dict[str, float],
        conflict_score: float,
        embodied_state: dict[str, float],
        config: "WorkspaceConfig",
    ) -> WorkspaceOutput:
        """Collect signals from all sources, rank, and broadcast top-N.

        Args:
            recalled_memories: Memories returned by retrieval.
            appraisal: AppraisalResult from amygdala.
            prediction_error: From predictive processor.
            mood: Current mood dimensions.
            conflict_score: From ACC / conflict monitor.
            embodied_state: Energy, cognitive load, comfort.
            config: WorkspaceConfig with thresholds.

        Returns:
            WorkspaceOutput with broadcast vs unconscious split.
        """
        signals: list[WorkspaceSignal] = []

        # Emotion signal from appraisal
        emotion = getattr(appraisal, "primary_emotion", "neutral")
        intensity = getattr(appraisal, "intensity", 0.0)
        signals.append(WorkspaceSignal(
            source="amygdala",
            content=emotion,
            salience=compute_salience("emotion", emotion, emotion_intensity=intensity),
            signal_type="emotion",
        ))

        # Memory signals
        for mem in recalled_memories:
            signals.append(WorkspaceSignal(
                source="memory",
                content=mem,
                salience=compute_salience("memory", mem),
                signal_type="memory",
            ))

        # Conflict signal
        if conflict_score > 0.0:
            signals.append(WorkspaceSignal(
                source="conflict",
                content=conflict_score,
                salience=compute_salience("conflict", conflict_score),
                signal_type="conflict",
            ))

        # Prediction error signal
        if prediction_error > 0.0:
            signals.append(WorkspaceSignal(
                source="prediction",
                content=prediction_error,
                salience=compute_salience(
                    "prediction", prediction_error,
                    prediction_error=prediction_error,
                ),
                signal_type="prediction",
            ))

        # Mood shift signal
        signals.append(WorkspaceSignal(
            source="mood",
            content=mood,
            salience=compute_salience("mood_shift", mood),
            signal_type="mood_shift",
        ))

        # Rank and select
        output = rank_and_select(
            signals,
            max_memories=config.max_context_memories,
            max_signals=config.max_signals,
            identity_override=config.identity_threat_override,
        )

        # Publish broadcast event
        self._bus.publish(
            WORKSPACE_BROADCAST,
            {
                "broadcast_count": len(output.broadcast),
                "unconscious_count": len(output.unconscious),
                "memory_count": len(output.broadcast_memories),
            },
        )

        logger.info(
            "Workspace broadcast: %d signals (%d memories), %d unconscious",
            len(output.broadcast),
            len(output.broadcast_memories),
            len(output.unconscious),
        )

        return output
