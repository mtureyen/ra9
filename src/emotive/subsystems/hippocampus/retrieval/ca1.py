"""CA1 comparator — context match + familiarity/recollection split.

After CA3 completes patterns, CA1 checks: does this match current
context? Produces two distinct signals:
  - Familiarity (perirhinal analog): "I've encountered this before"
  - Recollection (hippocampal analog): "I remember the specific episode"

Also detects source confusion when two top candidates have similar
content from different people.

Brain analog: CA1 pyramidal cells comparing CA3 output with EC input.
Sources: Nature s41467-017-02752-1, Yonelinas (2002) dual-process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from .context_vector import ContextVector
from .ca3 import CompletionCandidate


@dataclass
class ComparatorResult:
    """Output of CA1 comparison for a single memory."""

    candidate: CompletionCandidate
    context_match: float  # 0-1, how well it matches current context
    familiarity_score: float  # perirhinal: fast, context-free
    recollection_score: float  # hippocampal: slow, context-rich
    final_score: float  # combined score for ranking
    source_uncertain: bool = False  # source confusion flagged
    confused_with: str | None = None  # "might be [other person]"


@dataclass
class SourceConfusion:
    """Detected source confusion between two memories."""

    memory_a_id: uuid.UUID
    memory_b_id: uuid.UUID
    person_a: str | None
    person_b: str | None
    shared_content_similarity: float


@dataclass
class CA1Output:
    """Full output of the CA1 comparator stage."""

    results: list[ComparatorResult]
    source_confusions: list[SourceConfusion] = field(default_factory=list)
    # Global signals
    best_familiarity: float = 0.0
    best_recollection: float = 0.0
    tot_triggered: bool = False  # tip-of-the-tongue state


def compare_and_filter(
    candidates: list[CompletionCandidate],
    context: ContextVector,
    encoding_contexts: dict[uuid.UUID, list[float]] | None = None,
    *,
    context_reject_threshold: float = 0.15,
    familiarity_threshold: float = 0.4,
    recollection_threshold: float = 0.3,
    tot_lower: float = 0.25,
    source_confusion_threshold: float = 0.85,
) -> CA1Output:
    """Run CA1 comparison on completed candidates.

    Filters by context match, computes familiarity/recollection split,
    detects source confusion, and identifies TOT state.

    Args:
        candidates: From CA3 pattern completion.
        context: Current drifting context vector.
        encoding_contexts: Stored context vectors per memory ID.
        context_reject_threshold: Below this, reject as false completion.
        familiarity_threshold: Above this, familiarity signal fires.
        recollection_threshold: Above this, full recollection achieved.
        tot_lower: Below recollection but above this = TOT state.
        source_confusion_threshold: Content similarity above this between
            different-source candidates triggers source confusion.

    Returns:
        CA1Output with scored results, source confusions, and signals.
    """
    enc_contexts = encoding_contexts or {}
    results: list[ComparatorResult] = []

    for candidate in candidates:
        # Compute context match
        stored_ctx = enc_contexts.get(candidate.memory_id)
        if stored_ctx is not None:
            ctx_match = context.similarity_to(stored_ctx)
        else:
            # No stored context — can't compare, give neutral score
            ctx_match = 0.4

        # Reject if context match is too low (false completion)
        if ctx_match < context_reject_threshold and candidate.completion_score < 0.7:
            continue

        # Familiarity: based on activation level (how strongly it completed)
        # Fast, context-free signal — "have I encountered this before?"
        familiarity = min(candidate.completion_score, 1.0)

        # Recollection: based on context match + detail availability
        # Slower, context-rich signal — "I remember the specific episode"
        detail_factor = 1.0  # could be modulated by detail_retention
        recollection = ctx_match * 0.6 + candidate.completion_score * 0.3 + detail_factor * 0.1
        recollection = min(recollection, 1.0)

        # Final score: weighted combination
        final = candidate.completion_score * 0.4 + ctx_match * 0.3 + recollection * 0.3

        results.append(
            ComparatorResult(
                candidate=candidate,
                context_match=ctx_match,
                familiarity_score=familiarity,
                recollection_score=recollection,
                final_score=final,
            )
        )

    # Sort by final score
    results.sort(key=lambda r: r.final_score, reverse=True)

    # Detect source confusion (E2): top 2 have similar content but different people
    confusions = _detect_source_confusion(results, source_confusion_threshold)
    for confusion in confusions:
        # Mark the affected results as uncertain
        for r in results:
            mid = r.candidate.memory_id
            if mid == confusion.memory_a_id:
                r.source_uncertain = True
                r.confused_with = confusion.person_b
            elif mid == confusion.memory_b_id:
                r.source_uncertain = True
                r.confused_with = confusion.person_a

    # Compute global signals
    best_fam = max((r.familiarity_score for r in results), default=0.0)
    best_rec = max((r.recollection_score for r in results), default=0.0)

    # TOT detection: familiarity fires but recollection doesn't
    tot = (best_fam > familiarity_threshold and best_rec < recollection_threshold
           and best_rec > tot_lower)

    return CA1Output(
        results=results,
        source_confusions=confusions,
        best_familiarity=best_fam,
        best_recollection=best_rec,
        tot_triggered=tot,
    )


def _detect_source_confusion(
    results: list[ComparatorResult],
    threshold: float,
) -> list[SourceConfusion]:
    """Detect when top candidates have similar content from different sources."""
    confusions = []

    # Only check top 5 candidates
    top = results[:5]
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            # Different person tags? (exclude emotion and system tags)
            _NON_PERSON_TAGS = {
                "joy", "trust", "awe", "surprise", "sadness", "anger",
                "fear", "disgust", "neutral", "curiosity", "contempt",
                "episodic", "semantic", "procedural", "gist",
                "inner_speech", "contradiction", "divergence", "withheld",
                "behavioral_coaching", "dmn_flash", "dmn_reflection",
                "between_session", "releasing", "prospective",
                "concept_hub", "meta_memory", "retrieval_experience",
            }
            a_persons = {t for t in a.candidate.tags
                         if t.islower() and len(t) > 2 and t not in _NON_PERSON_TAGS}
            b_persons = {t for t in b.candidate.tags
                         if t.islower() and len(t) > 2 and t not in _NON_PERSON_TAGS}

            if a_persons and b_persons and not a_persons.intersection(b_persons):
                # Different people — check content similarity
                if (
                    a.candidate.embedding is not None
                    and b.candidate.embedding is not None
                ):
                    import numpy as np

                    sim = float(
                        np.dot(a.candidate.embedding, b.candidate.embedding)
                        / (
                            np.linalg.norm(a.candidate.embedding)
                            * np.linalg.norm(b.candidate.embedding)
                            + 1e-10
                        )
                    )
                    if sim > threshold:
                        confusions.append(
                            SourceConfusion(
                                memory_a_id=a.candidate.memory_id,
                                memory_b_id=b.candidate.memory_id,
                                person_a=next(iter(a_persons), None),
                                person_b=next(iter(b_persons), None),
                                shared_content_similarity=sim,
                            )
                        )

    return confusions
