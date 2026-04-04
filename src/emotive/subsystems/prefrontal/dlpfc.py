"""Retrieval strategy controller — dlPFC analog.

Analyzes input to decide HOW to search: by person, topic, emotion,
time, or broad. Each strategy uses different retrieval weights.

Brain analog: dorsolateral prefrontal cortex (dlPFC).
  - Strategic retrieval: "WHAT to look for"
  - Directs the retrieval search mode
Sources: PMC6675388, PMC5009205.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from emotive.subsystems.hippocampus.retrieval.concept_cells import PersonNodeCache

# Patterns for detecting retrieval query types
_TIME_PATTERNS = [
    re.compile(r"\b(last time|first time|when did|how long ago)\b", re.I),
    re.compile(r"\b(yesterday|last week|last month|recently)\b", re.I),
    re.compile(r"\b(before|after|earlier|initially|originally)\b", re.I),
    re.compile(r"\b(the first|the last|back when)\b", re.I),
]

_EMOTION_PATTERNS = [
    re.compile(r"\b(when i was|when you were|when .+ felt)\b", re.I),
    re.compile(r"\b(felt|feeling|sad|happy|angry|scared|afraid)\b", re.I),
    re.compile(r"\b(upset|excited|worried|anxious|calm|peaceful)\b", re.I),
]

_RECALL_PATTERNS = [
    re.compile(r"\b(do you remember|remember when|recall)\b", re.I),
    re.compile(r"\b(you told me|i told you|we talked about)\b", re.I),
    re.compile(r"\b(what did .+ say|what happened)\b", re.I),
    re.compile(r"\b(have you heard|do you know about)\b", re.I),
]


@dataclass
class StrategyWeights:
    """Retrieval weight profile for a strategy."""

    tag: float = 0.0
    embedding: float = 0.0
    recency: float = 0.0
    activation: float = 0.0
    mood_match: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "tag": self.tag,
            "embedding": self.embedding,
            "recency": self.recency,
            "activation": self.activation,
            "mood_match": self.mood_match,
        }


# Strategy weight profiles
STRATEGY_WEIGHTS: dict[str, StrategyWeights] = {
    "PERSON": StrategyWeights(
        tag=0.40, embedding=0.25, recency=0.15, activation=0.15, mood_match=0.05,
    ),
    "TEMPORAL": StrategyWeights(
        tag=0.15, embedding=0.20, recency=0.40, activation=0.15, mood_match=0.10,
    ),
    "EMOTION": StrategyWeights(
        tag=0.25, embedding=0.20, recency=0.10, activation=0.15, mood_match=0.30,
    ),
    "TOPIC": StrategyWeights(
        tag=0.20, embedding=0.40, recency=0.15, activation=0.15, mood_match=0.10,
    ),
    "BROAD": StrategyWeights(
        tag=0.15, embedding=0.25, recency=0.30, activation=0.20, mood_match=0.10,
    ),
}


@dataclass
class StrategyResult:
    """Output of strategy analysis."""

    strategy: str  # PERSON / TEMPORAL / EMOTION / TOPIC / BROAD
    weights: StrategyWeights
    detected_person: str | None = None
    is_recall_query: bool = False


def select_strategy(
    text: str,
    person_cache: PersonNodeCache,
) -> StrategyResult:
    """Analyze input and select retrieval strategy.

    Priority order:
    1. Person detected → PERSON strategy
    2. Time reference → TEMPORAL strategy
    3. Emotion reference → EMOTION strategy
    4. Recall query → TOPIC strategy (embedding-focused)
    5. Default → BROAD strategy

    Returns strategy name + weight profile + detected metadata.
    """
    is_recall = any(p.search(text) for p in _RECALL_PATTERNS)

    # 1. Check for known person name
    person = person_cache.detect_person(text)
    if person:
        return StrategyResult(
            strategy="PERSON",
            weights=STRATEGY_WEIGHTS["PERSON"],
            detected_person=person,
            is_recall_query=is_recall,
        )

    # 2. Check for time references
    if any(p.search(text) for p in _TIME_PATTERNS):
        return StrategyResult(
            strategy="TEMPORAL",
            weights=STRATEGY_WEIGHTS["TEMPORAL"],
            is_recall_query=is_recall,
        )

    # 3. Check for emotion references
    if any(p.search(text) for p in _EMOTION_PATTERNS):
        return StrategyResult(
            strategy="EMOTION",
            weights=STRATEGY_WEIGHTS["EMOTION"],
            is_recall_query=is_recall,
        )

    # 4. If explicit recall query, use topic-focused
    if is_recall:
        return StrategyResult(
            strategy="TOPIC",
            weights=STRATEGY_WEIGHTS["TOPIC"],
            is_recall_query=True,
        )

    # 5. Default: broad
    return StrategyResult(
        strategy="BROAD",
        weights=STRATEGY_WEIGHTS["BROAD"],
        is_recall_query=False,
    )
