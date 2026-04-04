"""Activation tracking — per-memory exponential decay.

Each memory has an activation level that decays over time since last
access. Recently recalled memories are "hot" — high activation, easy
to re-access. Old unaccessed memories are "cold."

Type-dependent time constants (τ):
  - Priming (just accessed): τ = 1 hour
  - Episodic: τ = 24 hours
  - Emotional episodic: τ = 48 hours (amygdala stabilization)
  - Semantic: τ = 168 hours (1 week)
  - Procedural: τ = 720 hours (1 month)

After 3τ: ~5% remaining. After 5τ: baseline.

Sources: Frontiers fpsyg.2018.00416, Kahana & Adler 2002.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


# Time constants in hours
TAU_PRIMING = 1.0
TAU_EPISODIC = 24.0
TAU_EMOTIONAL = 48.0
TAU_SEMANTIC = 168.0
TAU_PROCEDURAL = 720.0

# Baseline activation (floor)
BASELINE_ACTIVATION = 0.05


def get_tau(memory_type: str, emotional_intensity: float | None = None) -> float:
    """Get the decay time constant for a memory type.

    Emotional episodic memories decay slower due to amygdala
    stabilization of the memory trace.
    """
    if memory_type == "procedural":
        return TAU_PROCEDURAL
    if memory_type == "semantic":
        return TAU_SEMANTIC
    if memory_type == "episodic":
        if emotional_intensity and emotional_intensity > 0.5:
            return TAU_EMOTIONAL
        return TAU_EPISODIC
    # Default: episodic
    return TAU_EPISODIC


def compute_activation(
    base_activation: float,
    last_access: datetime | None,
    memory_type: str = "episodic",
    emotional_intensity: float | None = None,
    now: datetime | None = None,
) -> float:
    """Compute current activation from base + elapsed time.

    A(t) = max(baseline, base * exp(-elapsed / τ))

    If never accessed, returns base_activation unchanged.
    """
    if last_access is None:
        return base_activation

    if now is None:
        now = datetime.now(timezone.utc)

    elapsed_hours = (now - last_access).total_seconds() / 3600
    if elapsed_hours < 0:
        elapsed_hours = 0

    tau = get_tau(memory_type, emotional_intensity)
    decayed = base_activation * math.exp(-elapsed_hours / tau)
    return max(BASELINE_ACTIVATION, decayed)


def compute_retrieval_strengthening(
    base: float = 0.05,
    effort: float = 0.0,
    spacing_bonus: float = 0.0,
) -> float:
    """Compute how much to strengthen a memory after retrieval.

    Effortful retrieval strengthens more (testing effect, E28).
    Spaced retrieval strengthens more (spacing effect, E27).

    Sources: Roediger & Karpicke (2006), Bjork desirable difficulty.
    """
    effort_bonus = effort * 0.15  # up to 0.15 for max effort
    total = base + effort_bonus + spacing_bonus
    return min(total, 0.5)  # cap total strengthening per retrieval


def compute_spacing_bonus(timestamps: list[str] | list[datetime]) -> float:
    """Compute spacing bonus from retrieval timestamp history.

    Spaced retrievals (across sessions) strengthen more than
    massed retrievals (within one session).

    Returns 0.0 for massed, up to 0.25 for well-spaced.
    """
    if len(timestamps) < 2:
        return 0.0

    # Parse timestamps if strings
    parsed = []
    for ts in timestamps:
        if isinstance(ts, str):
            try:
                parsed.append(datetime.fromisoformat(ts))
            except (ValueError, TypeError):
                continue
        elif isinstance(ts, datetime):
            parsed.append(ts)

    if len(parsed) < 2:
        return 0.0

    parsed.sort()

    # Compute average interval in hours
    intervals = []
    for i in range(1, len(parsed)):
        interval_hours = (parsed[i] - parsed[i - 1]).total_seconds() / 3600
        intervals.append(interval_hours)

    avg_interval = sum(intervals) / len(intervals)

    # Spacing bonus: longer intervals = more strengthening
    if avg_interval < 1:  # within same session
        return 0.0
    elif avg_interval < 24:  # same day, different sessions
        return 0.1
    elif avg_interval < 168:  # within a week
        return 0.2
    else:  # weekly or more
        return 0.25


def compute_retrieval_effort(
    best_completion_score: float,
    competitor_count: int,
    iterations_used: int,
    max_iterations: int,
    tot_active: bool = False,
) -> float:
    """Compute retrieval effort metric.

    Hard retrieval (weak signal, many competitors, deep search)
    produces higher effort. Feeds into cognitive load and
    effortful strengthening.

    Brain analog: ACC effort monitoring.
    Source: Botvinick et al. (2001).
    """
    effort = 0.0

    # Weak signal = more effort
    effort += (1.0 - best_completion_score) * 0.3

    # Many competitors = more effort
    effort += min(competitor_count * 0.05, 0.2)

    # Deep search = more effort
    if max_iterations > 0:
        effort += (iterations_used / max_iterations) * 0.2

    # TOT is inherently effortful
    if tot_active:
        effort += 0.3

    return min(effort, 1.0)
