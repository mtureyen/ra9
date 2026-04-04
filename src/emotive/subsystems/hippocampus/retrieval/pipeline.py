"""Three-phase neural retrieval pipeline — the full Anamnesis engine.

Replaces one-shot cosine similarity with brain-correct retrieval:

  Phase 1 (Direct): embedding similarity + person-node + activation decay
  Phase 2 (Completion): CA3 attractor dynamics + spreading activation
  Phase 3 (Elaboration): context reinstatement + spontaneous overlap

Per exchange: all three phases run sequentially.
Across exchanges: Phase 3 results feed Phase 1 of next exchange
via RetrievalState.previously_recalled.

Brain analog: three-phase retrieval (PMC4792674).
  Phase 1: ~500ms direct cued retrieval
  Phase 2: ~500-1500ms pattern completion
  Phase 3: ~1500ms+ elaboration + context reinstatement
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

import numpy as np
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.db.queries.memory_queries import search_by_embedding
from emotive.logging import get_logger

from .dentate_gyrus import PatternSeparator
from .activation import (
    compute_activation,
    compute_retrieval_effort,
    compute_retrieval_strengthening,
    compute_spacing_bonus,
)
from .ca1 import CA1Output, compare_and_filter
from .context_vector import ContextVector
from .ca3 import CompletionCandidate, pattern_complete
from .concept_cells import PersonNodeCache
from .state import RetrievalState
from emotive.subsystems.prefrontal.dlpfc import StrategyResult, select_strategy

if TYPE_CHECKING:
    from emotive.config.schema import EmotiveConfig

logger = get_logger("retrieval.pipeline")


@dataclass
class RetrievalResult:
    """Full output of the three-phase retrieval pipeline."""

    # Memories that enter consciousness (workspace input)
    conscious: list[dict] = field(default_factory=list)
    # Memories 6-20: available for cascade + unconscious priming
    unconscious_pool: list[dict] = field(default_factory=list)
    # E5: Priming keywords from unconscious pool
    priming_words: set[str] = field(default_factory=set)
    # E18: Narrative arc (person/theme focused)
    narrative: str | None = None
    # E21: Prospective memory triggers
    prospective_triggers: list[str] = field(default_factory=list)
    # Retrieval metadata
    strategy: str = "BROAD"
    detected_person: str | None = None
    is_recall_query: bool = False
    # Dual signal
    familiarity_score: float = 0.0
    recollection_score: float = 0.0
    # TOT state
    tot_active: bool = False
    tot_partial_person: str | None = None
    tot_partial_emotion: str | None = None
    # Source confusion
    source_confusions: list[dict] = field(default_factory=list)
    # Effort
    effort: float = 0.0
    # How many iterations ran
    iterations_used: int = 0
    # Candidates considered
    total_candidates: int = 0


def run_retrieval(
    db_session: Session,
    query_text: str,
    query_embedding: list[float],
    retrieval_state: RetrievalState,
    person_cache: PersonNodeCache,
    separator: PatternSeparator,
    *,
    mood: dict[str, float] | None = None,
    prediction_error: float = 0.0,
    emotional_intensity: float = 0.0,
    conscious_limit: int = 5,
    unconscious_limit: int = 15,
    person_trust: float = 0.5,
    comfort: float = 0.5,
    embedding_service=None,
    conversation_id=None,
) -> RetrievalResult:
    """Run the full three-phase neural retrieval pipeline.

    This is the main entry point. Replaces the old
    association_cortex.recall() → recall_memories() path.

    Args:
        db_session: Active DB session.
        query_text: Raw user input text.
        query_embedding: Pre-computed 1024-dim embedding (shared).
        retrieval_state: Persistent state across exchanges.
        person_cache: Person-node cache (built at boot).
        separator: DG pattern separator.
        mood: Current mood (6 dimensions).
        prediction_error: From predictive processor.
        emotional_intensity: From fast appraisal.
        conscious_limit: Max memories for workspace (working memory).
        unconscious_limit: Max memories for unconscious pool.
        person_trust: Trust level for current person (0-1). Used for E23.
        comfort: Embodied comfort level (0-1). Used for E23.
        embedding_service: For E19 meta-memory encoding.
        conversation_id: For E19 meta-memory encoding.

    Returns:
        RetrievalResult with conscious memories, unconscious pool,
        and all retrieval metadata.
    """
    result = RetrievalResult()
    now = datetime.now(timezone.utc)

    # 0. Encoding-retrieval mode balance (E22)
    strategy_result = select_strategy(query_text, person_cache)
    result.strategy = strategy_result.strategy
    result.detected_person = strategy_result.detected_person
    result.is_recall_query = strategy_result.is_recall_query

    retrieval_state.compute_mode_balance(
        prediction_error=prediction_error,
        emotional_intensity=emotional_intensity,
        is_recall_query=strategy_result.is_recall_query,
    )

    # Update topic tracking
    topic_changed = strategy_result.detected_person != retrieval_state.topic_person
    retrieval_state.update_topic(strategy_result.detected_person, topic_changed)
    retrieval_state.strategy = strategy_result.strategy

    # Compute theta iterations (deeper with sustained attention)
    theta_iters = retrieval_state.get_theta_iterations()
    result.iterations_used = theta_iters

    # 1. DG Pattern Separation
    separated_embedding = separator.separate(
        query_embedding,
        detected_person=strategy_result.detected_person,
    )

    # 2. Update context vector
    retrieval_state.context.drift(query_embedding)

    # 3. Get person-node memories (bypasses embedding)
    person_ids: set[uuid.UUID] | None = None
    if strategy_result.detected_person:
        person_ids = person_cache.get_memory_ids(strategy_result.detected_person)

    # 4. Get active RIF suppressions
    active_rif = retrieval_state.get_active_rif()

    # 5. Compute mood valence for congruent boosting
    mood_valence: float | None = None
    if mood:
        mood_valence = sum(mood.values()) / max(len(mood), 1)

    # LC: Locus coeruleus scope modulation
    from emotive.subsystems.locus_coeruleus import compute_retrieval_scope
    lc_scope = compute_retrieval_scope(
        prediction_error=prediction_error,
        emotional_intensity=emotional_intensity,
        energy=getattr(retrieval_state, '_energy', 1.0),
    )
    effective_seed_limit = int(
        (conscious_limit + unconscious_limit) * lc_scope["candidate_multiplier"]
    )

    # === PHASE 1 + 2: CA3 Pattern Completion ===
    # (Combines direct retrieval + attractor dynamics)
    candidates = pattern_complete(
        db_session,
        separated_embedding.tolist() if isinstance(separated_embedding, np.ndarray) else separated_embedding,
        iterations=theta_iters,
        seed_limit=effective_seed_limit,
        person_memory_ids=person_ids,
        exclude_ids=set(),
        rif_suppressed=active_rif,
        current_mood_valence=mood_valence,
    )

    result.total_candidates = len(candidates)

    if not candidates:
        # Failed retrieval — set deferred activation (E1)
        _handle_failed_retrieval(db_session, retrieval_state, query_embedding)
        return result

    # === PHASE 3: CA1 Context Comparison ===
    # Load encoding contexts for context matching
    encoding_contexts = _load_encoding_contexts(
        db_session,
        [c.memory_id for c in candidates[:20]],
    )

    ca1_output = compare_and_filter(
        candidates,
        retrieval_state.context,
        encoding_contexts=encoding_contexts,
    )

    # Update retrieval state with CA1 signals
    retrieval_state.last_familiarity_score = ca1_output.best_familiarity
    retrieval_state.last_recollection_score = ca1_output.best_recollection
    result.familiarity_score = ca1_output.best_familiarity
    result.recollection_score = ca1_output.best_recollection

    # TOT state (E11)
    if ca1_output.tot_triggered:
        result.tot_active = True
        # Extract partial features from near-miss
        if ca1_output.results:
            top = ca1_output.results[0]
            result.tot_partial_person = next(
                (t for t in top.candidate.tags if t.islower() and len(t) > 2),
                None,
            )
            result.tot_partial_emotion = top.candidate.primary_emotion
        # Feed into deferred retrieval
        retrieval_state.tot = retrieval_state.tot.__class__(
            active=True,
            partial_person=result.tot_partial_person,
            partial_emotion=result.tot_partial_emotion,
            near_miss_ids=[r.candidate.memory_id for r in ca1_output.results[:3]],
            confidence=ca1_output.best_familiarity,
        )

    # Source confusions (E2)
    result.source_confusions = [
        {
            "person_a": sc.person_a,
            "person_b": sc.person_b,
            "similarity": sc.shared_content_similarity,
        }
        for sc in ca1_output.source_confusions
    ]

    # E12: Proactive interference — old high-activation memories blocking newer ones
    from .interference import detect_proactive_interference
    ca1_output.results = detect_proactive_interference(ca1_output.results)

    # dlPFC: Apply strategy weights to scoring
    # Human brain: HOW you search affects WHAT you find. Person queries
    # weight tag matches higher. Temporal queries weight recency higher.
    weights = strategy_result.weights
    for r in ca1_output.results:
        c = r.candidate
        tag_bonus = 0.0
        recency_bonus = 0.0

        # Tag relevance (person name, emotion, theme)
        if strategy_result.detected_person:
            if strategy_result.detected_person in [t.lower() for t in c.tags]:
                tag_bonus = weights.tag * 0.3

        # Recency (recent memories score higher for TEMPORAL strategy)
        if c.created_at:
            age_days = (now - c.created_at).total_seconds() / 86400
            recency_bonus = weights.recency * max(0, 0.2 - age_days * 0.01)

        r.final_score += tag_bonus + recency_bonus

    # E14: Somatic marker biases from body state
    from emotive.subsystems.insula.somatic_markers import compute_somatic_bias, apply_somatic_bias_to_score
    somatic_bias = compute_somatic_bias(
        energy=getattr(retrieval_state, '_energy', 1.0),
        cognitive_load=getattr(retrieval_state, '_cognitive_load', 0.0),
        comfort=comfort,
    )

    # === Apply score modifiers (E6, E10, E12, E13, E14, E20, E24, E30, E31) ===
    from .suppression import get_effective_suppression
    from .resistance import compute_resistance_threshold
    from emotive.subsystems.basal_ganglia import get_gating_bonus

    for r in ca1_output.results:
        c = r.candidate

        # E6: Reminiscence bump — formation period memories get +0.15
        if c.metadata.get("formation_period") or getattr(c, 'is_formative', False):
            r.final_score += 0.15

        # E10: Mood-dependent gating — encoding mood match
        if mood and c.metadata.get("encoding_mood"):
            enc_mood = c.metadata["encoding_mood"]
            if isinstance(enc_mood, dict) and enc_mood:
                # Cosine-like: compare current mood dimensions to encoding mood
                shared_dims = set(mood.keys()) & set(enc_mood.keys())
                if shared_dims:
                    dot = sum(mood[d] * enc_mood.get(d, 0.5) for d in shared_dims)
                    mag_a = sum(mood[d] ** 2 for d in shared_dims) ** 0.5
                    mag_b = sum(enc_mood.get(d, 0.5) ** 2 for d in shared_dims) ** 0.5
                    if mag_a > 0 and mag_b > 0:
                        mood_match = dot / (mag_a * mag_b)
                        r.final_score += mood_match * 0.15

        # E13: Intentional suppression — check suppression_level
        supp_level = c.metadata.get("suppression_level") or 0.0
        supp_start = c.metadata.get("suppression_decay_start")
        if supp_level > 0:
            import math
            start_dt = None
            if supp_start:
                try:
                    start_dt = datetime.fromisoformat(str(supp_start))
                except (ValueError, TypeError):
                    pass
            effective_supp = get_effective_suppression(
                supp_level, start_dt,
                energy=getattr(retrieval_state, '_energy', 1.0),
            )
            # E14: Somatic marker further weakens suppression
            if somatic_bias.suppression_weakening > 0:
                effective_supp *= (1.0 - somatic_bias.suppression_weakening)
            r.final_score *= (1.0 - effective_supp)

        # E20: Memory resistance — identity-threatening memories harder to reach
        identity_resistance = c.metadata.get("identity_resistance", 0)
        if identity_resistance > 0:
            threshold_mult = compute_resistance_threshold(
                identity_resistance,
                person_trust=person_trust,
                energy=getattr(retrieval_state, '_energy', 0.5),
                comfort=comfort,
                is_direct_inquiry=strategy_result.is_recall_query,
            )
            r.final_score /= threshold_mult  # higher threshold = lower score

        # E24: Pre-architecture amnesia — memories without encoding_mood get penalty
        if not c.metadata.get("encoding_mood") and c.emotional_intensity is None:
            r.final_score *= 0.75
            if c.is_formative and c.retrieval_count > 10:
                r.final_score *= 1.2

        # E30: Von Restorff — high encoding PE = less retrieval competition
        encoding_pe = c.metadata.get("encoding_prediction_error", 0)
        if encoding_pe > 0.5:
            r.final_score *= (1.0 + encoding_pe * 0.2)

        # E31: Reward-based gating — useful memories get priority
        gating = get_gating_bonus(c.metadata)
        r.final_score += gating

        # E14: Somatic marker bias — body state shapes memory access
        r.final_score = apply_somatic_bias_to_score(
            r.final_score,
            memory_tags=c.tags,
            memory_type=c.memory_type,
            primary_emotion=c.primary_emotion,
            bias=somatic_bias,
        )

    # Re-sort after all modifiers
    ca1_output.results.sort(key=lambda r: r.final_score, reverse=True)

    # === Split into conscious + unconscious ===
    all_scored = ca1_output.results

    # E14: Somatic retrieval narrowing — high cognitive load filters weak signals
    if somatic_bias.retrieval_narrowing:
        # Only keep memories above median score (strongest signals only)
        if len(all_scored) > 2:
            scores = [r.final_score for r in all_scored]
            median_score = sorted(scores)[len(scores) // 2]
            all_scored = [r for r in all_scored if r.final_score >= median_score]

    # Apply within-retrieval inhibition (E4): sequential suppression
    inhibited = _apply_within_retrieval_inhibition(all_scored)

    # Conscious: top N (workspace input)
    conscious_candidates = inhibited[:conscious_limit]
    # Unconscious pool: next M
    unconscious_candidates = inhibited[conscious_limit : conscious_limit + unconscious_limit]

    # Convert to memory dicts (compatible with workspace/PFC)
    result.conscious = [_candidate_to_dict(r, now) for r in conscious_candidates]
    result.unconscious_pool = [_candidate_to_dict(r, now) for r in unconscious_candidates]

    # E5: Extract unconscious priming keywords
    result.priming_words = _extract_priming_keywords(
        [r.candidate for r in unconscious_candidates],
        [r.candidate for r in conscious_candidates],
    )

    # E18: Narrative reconstruction (person/theme focused)
    from .narrative import construct_narrative
    narrative = construct_narrative(
        result.conscious,
        detected_person=strategy_result.detected_person,
    )
    result.narrative = narrative

    # Spontaneous recall: context overlap check (replaces 5% random DMN flash)
    spontaneous = _check_spontaneous_recall(
        db_session, retrieval_state.context,
        exclude_ids={r.candidate.memory_id for r in conscious_candidates}
        | {r.candidate.memory_id for r in unconscious_candidates},
        somatic_bias=somatic_bias,
    )
    if spontaneous:
        result.conscious.append(spontaneous)

    # E21: Prospective memory triggers
    from .prospective import check_prospective_triggers
    prospective_triggers = check_prospective_triggers(
        getattr(retrieval_state, '_prospective_cache', []),
        query_text,
        detected_person=strategy_result.detected_person,
    )
    result.prospective_triggers = prospective_triggers

    # === Post-retrieval updates ===

    # Compute effort (E15)
    best_score = ca1_output.results[0].final_score if ca1_output.results else 0.0
    competitor_count = len([r for r in ca1_output.results if r.final_score > best_score * 0.7])
    result.effort = compute_retrieval_effort(
        best_completion_score=best_score,
        competitor_count=competitor_count,
        iterations_used=theta_iters,
        max_iterations=5,
        tot_active=result.tot_active,
    )
    retrieval_state.last_effort = result.effort

    # Update previously_recalled for next exchange cascade
    retrieval_state.previously_recalled = [
        r.candidate.memory_id for r in conscious_candidates
    ]
    retrieval_state.unconscious_pool = [
        r.candidate.memory_id for r in unconscious_candidates
    ]

    # Apply RIF to competitors (E4 cross-exchange)
    for r in conscious_candidates:
        for other in all_scored:
            if other.candidate.memory_id == r.candidate.memory_id:
                continue
            if other.candidate.embedding is not None and r.candidate.embedding is not None:
                overlap = float(
                    np.dot(other.candidate.embedding, r.candidate.embedding)
                    / (
                        np.linalg.norm(other.candidate.embedding)
                        * np.linalg.norm(r.candidate.embedding)
                        + 1e-10
                    )
                )
                if overlap > 0.7:
                    retrieval_state.add_rif(other.candidate.memory_id, suppression=0.3)

    # Mark retrieved memories: update retrieval stats, lability, strengthening, drift
    _mark_retrieved(
        db_session,
        [r.candidate.memory_id for r in conscious_candidates],
        effort=result.effort,
        now=now,
        context_vector=retrieval_state.context,
        person_trust=person_trust,
        comfort=comfort,
        prediction_error=prediction_error,
        detected_person=result.detected_person,
    )

    # E19: Meta-memory encoding — remembering what it felt like to remember
    _maybe_encode_meta_memory(
        db_session,
        conscious_candidates,
        effort=result.effort,
        emotional_intensity=emotional_intensity,
        tot_resolved=result.tot_active,
        embedding_service=embedding_service,
        conversation_id=conversation_id,
        retrieval_state=retrieval_state,
    )

    logger.info(
        "Retrieval: strategy=%s, candidates=%d, conscious=%d, effort=%.2f, tot=%s",
        result.strategy,
        result.total_candidates,
        len(result.conscious),
        result.effort,
        result.tot_active,
    )

    return result


def _apply_within_retrieval_inhibition(
    results: list,
) -> list:
    """Sequential suppression within a single retrieval attempt (E4).

    The first item that surfaces suppresses competitors with high overlap.
    Each winner suppresses remaining candidates for next iteration.
    """
    if len(results) <= 1:
        return results

    selected = []
    remaining = list(results)

    while remaining and len(selected) < len(results):
        # Best remaining candidate
        winner = remaining.pop(0)
        selected.append(winner)

        # Suppress remaining candidates that overlap with winner
        if winner.candidate.embedding is not None:
            for other in remaining:
                if other.candidate.embedding is not None:
                    overlap = float(
                        np.dot(winner.candidate.embedding, other.candidate.embedding)
                        / (
                            np.linalg.norm(winner.candidate.embedding)
                            * np.linalg.norm(other.candidate.embedding)
                            + 1e-10
                        )
                    )
                    if overlap > 0.6:
                        # Suppress: reduce score
                        suppression = 0.4 * overlap
                        other.final_score *= (1.0 - suppression)

            # Re-sort remaining by suppressed scores
            remaining.sort(key=lambda r: r.final_score, reverse=True)

    return selected


def _handle_failed_retrieval(
    db_session: Session,
    state: RetrievalState,
    query_embedding: list[float],
) -> None:
    """Handle failed retrieval — set up deferred activation (E1).

    Failed retrieval sensitizes the brain to relevant cues.
    Near-miss memories get a deferred_activation boost that persists
    and makes them easier to find next time.
    """
    from sqlalchemy import update

    near_miss_ids = list(state.tot.near_miss_ids) if state.tot.active else []
    if not near_miss_ids:
        return

    from sqlalchemy import func as sa_func

    for mid in near_miss_ids:
        db_session.execute(
            update(Memory)
            .where(Memory.id == mid)
            .values(
                deferred_activation=sa_func.least(
                    Memory.__table__.c.deferred_activation + 0.3, 1.0
                )
            )
        )
    db_session.flush()
    logger.info("Deferred activation set for %d near-miss memories", len(near_miss_ids))


def _load_encoding_contexts(
    db_session: Session,
    memory_ids: list[uuid.UUID],
) -> dict[uuid.UUID, list[float]]:
    """Load stored encoding context vectors for context matching.

    Currently returns empty dict since encoding_mood is JSONB (mood state),
    not a full context vector. The context vector for encoding will be
    populated going forward.
    """
    # TODO: Once memories are encoded with context vector snapshots,
    # load them here. For now, CA1 uses neutral context match (0.4).
    return {}


def _candidate_to_dict(result, now: datetime) -> dict:
    """Convert a ComparatorResult to the dict format expected by workspace/PFC."""
    c = result.candidate
    return {
        "id": c.memory_id,
        "memory_type": c.memory_type,
        "content": c.content,
        "tags": c.tags,
        "metadata": c.metadata,
        "confidence": 1.0,
        "reinforcement_count": 0,
        "detail_retention": 1.0,
        "is_formative": c.is_formative,
        "retrieval_count": c.retrieval_count,
        "created_at": c.created_at,
        "is_labile": True,
        "primary_emotion": c.primary_emotion,
        "similarity": c.raw_similarity,
        "recency_weight": 0.0,  # computed by old system, not needed here
        "spreading_activation": 0.0,
        "significance": c.metadata.get("significance", 0.5),
        "final_rank": result.final_score,
        # New Phase Anamnesis fields
        "familiarity_score": result.familiarity_score,
        "recollection_score": result.recollection_score,
        "context_match": result.context_match,
        "source_uncertain": result.source_uncertain,
        "confused_with": result.confused_with,
    }


def _mark_retrieved(
    db_session: Session,
    memory_ids: list[uuid.UUID],
    effort: float,
    now: datetime,
    context_vector: ContextVector | None = None,
    person_trust: float = 0.5,
    comfort: float = 0.5,
    prediction_error: float = 0.0,
    detected_person: str | None = None,
) -> None:
    """Mark memories as retrieved — update stats, lability, strengthening, drift.

    Implements:
      E1: Consume deferred_activation (reset to 0)
      E3: Source strength decay (3x faster than content)
      E9: Retrieval-induced drift (blend embedding with context)
      E17: Collaborative memory tagging (co_constructed_with)
      E23: Emotional blunting through safe recall / sensitization
      E25: Full reconsolidation — prediction error gates plasticity
      E27: Append retrieval timestamp for spacing effect
      E28: Effortful retrieval strengthening
    """
    import json
    import math

    from sqlalchemy import select, update

    if not memory_ids:
        return

    labile_until = now + timedelta(hours=1)

    for mid in memory_ids:
        # Load memory for per-memory computations
        mem = db_session.get(Memory, mid)
        if mem is None:
            continue

        # E27: Compute spacing bonus from timestamps
        timestamps = mem.retrieval_timestamps or []
        spacing = compute_spacing_bonus(timestamps)

        # E28: Effortful strengthening
        strengthening = compute_retrieval_strengthening(
            base=0.05, effort=effort, spacing_bonus=spacing,
        )

        # E27: Append current timestamp
        timestamps.append(now.isoformat())
        # Keep last 20 timestamps
        if len(timestamps) > 20:
            timestamps = timestamps[-20:]

        # E3: Source amnesia — decay source_strength (3x faster than content)
        source_decay = 0.0003
        if mem.last_access_at:
            hours_since = (now - mem.last_access_at).total_seconds() / 3600
            new_source = (mem.source_strength or 1.0) * math.exp(-hours_since * source_decay)
            new_source = max(new_source, 0.05)
        else:
            new_source = mem.source_strength or 1.0

        # E9: Retrieval-induced drift — blend embedding with current context
        if (context_vector is not None
                and mem.embedding is not None
                and (mem.retrieval_count or 0) > 5):
            drift_factor = min((mem.retrieval_count or 0) * 0.002, 0.15)
            # Emotional memories drift less
            if mem.emotional_intensity and mem.emotional_intensity > 0.5:
                drift_factor *= (1 - mem.emotional_intensity * 0.5)
            # Formative memories drift least
            if mem.is_formative:
                drift_factor *= 0.3

            # E9 drift: use only the first context.dim dimensions of embedding
            # to avoid zero-padding distortion (context=256, embedding=1024)
            ctx_snap = context_vector.snapshot()
            orig = list(mem.embedding)
            ctx_dim = len(ctx_snap)
            emb_dim = len(orig)

            # Only drift the first ctx_dim dimensions (where context has signal)
            # Leave remaining dimensions unchanged
            drifted = list(orig)  # copy
            for i in range(min(ctx_dim, emb_dim)):
                drifted[i] = (1 - drift_factor) * orig[i] + drift_factor * ctx_snap[i]

            # Normalize to preserve original magnitude
            orig_norm = sum(x * x for x in orig) ** 0.5
            drift_norm = sum(x * x for x in drifted) ** 0.5
            if drift_norm > 1e-10 and orig_norm > 1e-10:
                scale = orig_norm / drift_norm
                drifted = [x * scale for x in drifted]
            mem.embedding = drifted

        # E23: Emotional blunting through safe recall
        # Humans: safe recall → reconsolidate with less emotion.
        # Unsafe recall (genuine threat) → sensitization.
        # Neutral recall → no change. The brain doesn't sensitize on neutral.
        if mem.is_labile:
            ei = mem.emotional_intensity or 0.0
            if person_trust > 0.6 and comfort > 0.5:
                # Safe conditions: blunt emotional intensity (exposure therapy)
                floor = 0.5 if mem.is_formative else 0.3
                ei = max(ei * 0.97, floor)
            elif person_trust < 0.3 or comfort < 0.3:
                # Genuinely unsafe: slight sensitization (re-traumatization)
                ei = min(ei * 1.02, 1.0)
            # else: neutral conditions → no change (human-correct)
            mem.emotional_intensity = ei

        # E25: Full reconsolidation — prediction error gates plasticity
        # Human brain: routine recall strengthens. Surprising recall opens
        # the memory for modification. Compatible new info integrates.
        # Contradictory info weakens. Too surprising → new encoding.
        if prediction_error < 0.2:
            # Routine recall: just strengthen (already done via strengthening above)
            pass
        elif prediction_error < 0.6:
            # Moderate surprise: TRUE reconsolidation window open
            # Check compatibility: is the current context compatible with this memory?
            # Use embedding similarity as proxy for content compatibility
            if mem.embedding is not None and context_vector is not None:
                ctx_snap = context_vector.snapshot()
                emb_slice = list(mem.embedding)[:len(ctx_snap)]
                import numpy as _np
                _emb = _np.asarray(emb_slice, dtype=_np.float32)
                _ctx = _np.asarray(ctx_snap, dtype=_np.float32)
                _n1 = _np.linalg.norm(_emb)
                _n2 = _np.linalg.norm(_ctx)
                compatibility = float(_np.dot(_emb, _ctx) / (_n1 * _n2 + 1e-10)) if _n1 > 0 and _n2 > 0 else 0.5

                if compatibility > 0.4:
                    # Compatible: integrate (strengthen + tags handled in base.py)
                    strengthening *= 1.2  # bonus for compatible reconsolidation
                else:
                    # Contradictory: weaken
                    mem.confidence = (mem.confidence or 1.0) * 0.8
                    activation_now = mem.current_activation or 0.5
                    strengthening = activation_now * 0.7 - activation_now  # net negative
            else:
                # Can't assess compatibility — default to mild weakening
                mem.confidence = (mem.confidence or 1.0) * 0.9
        else:
            # High surprise (PE >= 0.6): new encoding, don't modify old
            strengthening = 0.0

        # E17: Collaborative memory tagging
        if detected_person:
            meta = dict(mem.metadata_ or {})
            co_constructed = meta.get("co_constructed_with", {})
            if not isinstance(co_constructed, dict):
                co_constructed = {}
            co_constructed[detected_person] = co_constructed.get(detected_person, 0) + 1
            meta["co_constructed_with"] = co_constructed
            mem.metadata_ = meta

        # Update all fields
        mem.retrieval_count = (mem.retrieval_count or 0) + 1
        mem.last_retrieved = now
        mem.last_access_at = now
        mem.is_labile = True
        mem.labile_until = labile_until
        mem.current_activation = max(0.05, (mem.current_activation or 0.5) + strengthening)
        mem.retrieval_timestamps = timestamps
        mem.source_strength = new_source
        # E1: Consume deferred activation
        mem.deferred_activation = 0.0

    db_session.flush()


def _check_spontaneous_recall(
    db_session: Session,
    context: ContextVector,
    exclude_ids: set[uuid.UUID] | None = None,
    sample_size: int = 50,
    threshold: float = 0.35,
    somatic_bias=None,
) -> dict | None:
    """Spontaneous recall — context overlap without explicit cue.

    Samples random memories and checks if their encoding context
    overlaps with the current context vector. If overlap > threshold,
    the memory surfaces spontaneously.

    Replaces the 5% random DMN flash with context-driven activation.

    Brain analog: DMN spontaneous retrieval from context overlap.
    Sources: PMC7741080, PLOS pbio.3003258.
    """
    from sqlalchemy import func as sa_func, select

    exclude = exclude_ids or set()

    # Sample random active memories
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(Memory.embedding.isnot(None))
        .order_by(sa_func.random())
        .limit(sample_size)
    )
    rows = db_session.execute(stmt).scalars().all()

    now = datetime.now(timezone.utc)
    best_overlap = 0.0
    best_memory = None

    for m in rows:
        if m.id in exclude:
            continue

        # Check context overlap using encoding_mood as proxy
        # (full context vectors not yet stored at encoding)
        # Use embedding similarity to context snapshot as approximation
        if m.embedding is not None:
            # Use first context.dim dimensions of embedding for comparison.
            # This is approximate but avoids zero-padding in the other direction.
            # The context vector captures conversational drift in 256-dim;
            # we compare against the same subspace of the 1024-dim embedding.
            emb_list = list(m.embedding)
            overlap = context.similarity_to(emb_list[:context.dim])

            # Emotional memories 2x easier to spontaneously surface
            # (amygdala-tagged memories have stronger context overlap)
            if m.emotional_intensity and m.emotional_intensity > 0.5:
                overlap *= 2.0

            # Recent memories 3x more likely (recency bias in spontaneous recall)
            if m.created_at:
                age_hours = (now - m.created_at).total_seconds() / 3600
                if age_hours < 24:
                    overlap *= 2.0  # very recent
                elif age_hours < 168:
                    overlap *= 1.5  # within a week

            # E14: Low energy lowers spontaneous threshold (defenses down)
            if somatic_bias and somatic_bias.spontaneous_threshold_reduction > 0:
                overlap += somatic_bias.spontaneous_threshold_reduction

            if overlap > threshold and overlap > best_overlap:
                best_overlap = overlap
                best_memory = m

    if best_memory is None:
        return None

    logger.info(
        "Spontaneous recall: %.2f overlap — %s",
        best_overlap, best_memory.content[:60],
    )

    return {
        "id": best_memory.id,
        "memory_type": best_memory.memory_type,
        "content": best_memory.content,
        "tags": best_memory.tags or [],
        "metadata": best_memory.metadata_ or {},
        "confidence": best_memory.confidence or 1.0,
        "reinforcement_count": best_memory.reinforcement_count or 0,
        "detail_retention": best_memory.detail_retention or 1.0,
        "is_formative": best_memory.is_formative or False,
        "retrieval_count": best_memory.retrieval_count or 0,
        "created_at": best_memory.created_at,
        "is_labile": False,
        "primary_emotion": best_memory.primary_emotion,
        "similarity": best_overlap,
        "final_rank": best_overlap,
        "spontaneous": True,  # flag for PFC context
    }


def _extract_priming_keywords(
    unconscious: list,
    conscious: list,
    keywords_per_memory: int = 3,
) -> set[str]:
    """E5: Extract priming keywords from the unconscious pool.

    These words subtly influence the LLM's response without being
    explicit recalled content. The unconscious pool activates semantic
    networks below awareness.

    Brain analog: semantic priming via spreading activation.
    """
    import re

    STOP_WORDS = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "it", "its", "i", "me",
        "my", "you", "your", "he", "she", "they", "them", "we", "us", "this",
        "that", "these", "those", "what", "which", "who", "whom", "how",
        "not", "no", "nor", "and", "but", "or", "so", "if", "than", "too",
        "very", "just", "about", "also", "more", "some", "any", "all",
    }

    conscious_words = set()
    for c in conscious:
        words = re.findall(r'\b[a-z]{3,}\b', c.content.lower())
        conscious_words.update(words)

    priming = set()
    for c in unconscious:
        words = re.findall(r'\b[a-z]{3,}\b', c.content.lower())
        distinctive = [w for w in words if w not in STOP_WORDS and w not in conscious_words]
        priming.update(distinctive[:keywords_per_memory])

    return priming


def _maybe_encode_meta_memory(
    db_session: Session,
    conscious_candidates: list,
    *,
    effort: float,
    emotional_intensity: float,
    tot_resolved: bool,
    embedding_service,
    conversation_id,
    retrieval_state: RetrievalState,
) -> None:
    """E19: Meta-memory encoding — remembering what it felt like to remember.

    After significant retrievals (high emotion > 0.5, high effort > 0.6,
    or TOT resolved), store a meta-memory: "Remembered X. It felt Y."
    Cap at 1 per session.

    Brain analog: metamemory — the subjective experience of remembering.
    """
    if embedding_service is None:
        return

    # Cap: only 1 meta-memory per session
    if getattr(retrieval_state, '_meta_memory_stored', False):
        return

    # Check significance thresholds
    is_significant = (
        emotional_intensity > 0.5
        or effort > 0.6
        or tot_resolved
    )
    if not is_significant or not conscious_candidates:
        return

    top = conscious_candidates[0]
    content_snippet = top.candidate.content[:100] if hasattr(top, 'candidate') else str(top)[:100]

    # Build the meta-memory description
    feelings = []
    if effort > 0.6:
        feelings.append("effortful")
    if emotional_intensity > 0.5:
        feelings.append("emotionally charged")
    if tot_resolved:
        feelings.append("a tip-of-tongue moment resolved")
    feeling_str = ", ".join(feelings) if feelings else "significant"

    meta_content = f"Remembered: '{content_snippet}'. It felt {feeling_str}."

    try:
        from emotive.memory.episodic import store_episodic
        store_episodic(
            db_session, embedding_service,
            content=meta_content,
            conversation_id=conversation_id,
            tags=["meta_memory", "retrieval_experience"],
        )
        retrieval_state._meta_memory_stored = True
        logger.info("Meta-memory encoded: %s", meta_content[:80])
    except Exception:
        logger.exception("Failed to encode meta-memory")
