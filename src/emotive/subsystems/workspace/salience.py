"""Salience computation and signal ranking for the global workspace.

Pure functions -- no side effects, no event bus, no DB.
"""

from __future__ import annotations

from .signals import WorkspaceOutput, WorkspaceSignal


def compute_salience(
    signal_type: str,
    content: object,
    emotion_intensity: float = 0.0,
    prediction_error: float = 0.0,
) -> float:
    """Compute salience for a signal based on its type and context.

    Returns:
        Salience score in [0, 1].
    """
    if signal_type == "emotion":
        return max(0.0, min(1.0, emotion_intensity))

    if signal_type == "memory":
        # Memory salience = retrieval_score (similarity)
        if isinstance(content, dict):
            return max(0.0, min(1.0, float(content.get("retrieval_score", 0.0))))
        return 0.0

    if signal_type == "conflict":
        # Identity threats are boosted 1.5x
        score = float(content) if isinstance(content, (int, float)) else 0.0
        return max(0.0, min(1.0, score * 1.5))

    if signal_type == "prediction":
        return max(0.0, min(1.0, prediction_error))

    if signal_type == "mood_shift":
        # Deviation from baseline (0.5)
        if isinstance(content, dict):
            deviations = [abs(v - 0.5) for v in content.values() if isinstance(v, (int, float))]
            return max(0.0, min(1.0, max(deviations) if deviations else 0.0))
        return 0.0

    return 0.0


def rank_and_select(
    signals: list[WorkspaceSignal],
    max_memories: int = 5,
    max_signals: int = 8,
    identity_override: bool = True,
) -> WorkspaceOutput:
    """Rank signals by salience and split into broadcast vs unconscious.

    Identity threats (conflict signals) always broadcast if identity_override
    is True, regardless of rank.

    Returns:
        WorkspaceOutput with broadcast, unconscious, and broadcast_memories.
    """
    if not signals:
        return WorkspaceOutput()

    # Separate identity threats if override is on
    forced: list[WorkspaceSignal] = []
    rest: list[WorkspaceSignal] = []

    for sig in signals:
        if identity_override and sig.signal_type == "conflict" and sig.salience > 0.0:
            forced.append(sig)
        else:
            rest.append(sig)

    # Sort rest by salience descending
    rest.sort(key=lambda s: s.salience, reverse=True)

    # Build broadcast list: forced first, then top-N from rest
    broadcast: list[WorkspaceSignal] = list(forced)
    slots_remaining = max(0, max_signals - len(broadcast))
    broadcast.extend(rest[:slots_remaining])

    # Unconscious: everything not broadcast
    broadcast_set = set(id(s) for s in broadcast)
    unconscious = [s for s in signals if id(s) not in broadcast_set]

    # Extract broadcast memories (from memory-type signals that made it in)
    broadcast_memories: list[dict] = []
    memory_count = 0
    for sig in broadcast:
        if sig.signal_type == "memory" and isinstance(sig.content, dict):
            if memory_count < max_memories:
                broadcast_memories.append(sig.content)
                memory_count += 1

    return WorkspaceOutput(
        broadcast=broadcast,
        unconscious=unconscious,
        broadcast_memories=broadcast_memories,
    )
