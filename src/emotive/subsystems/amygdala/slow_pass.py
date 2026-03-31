"""Slow-pass appraisal: post-LLM, full exchange analysis with reappraisal.

Runs on the complete exchange (user message + LLM response) AFTER the
response is generated. Compares the exchange embedding against complex
situation prototypes. Can override the fast pass (reappraisal).

Brain analog: cortical appraisal via right TPJ + IFG (200-600ms).
"""

from __future__ import annotations

import math

from emotive.embeddings.service import EmbeddingService
from emotive.layers.appraisal import AppraisalResult, AppraisalVector

from .fast_pass import EMOTION_TO_APPRAISAL, cosine_similarity


# Mapping from situation prototypes to their emotional profiles.
# These are more nuanced than raw emotion prototypes.
SITUATION_TO_APPRAISAL: dict[str, dict[str, float]] = {
    "vulnerability_sharing":  {"valence": 0.4,  "goal_relevance": 0.8, "novelty": 0.6, "agency": 0.3, "social_significance": 0.9},
    "trust_deepening":        {"valence": 0.8,  "goal_relevance": 0.8, "novelty": 0.4, "agency": 0.5, "social_significance": 0.9},
    "playful_teasing":        {"valence": 0.75, "goal_relevance": 0.3, "novelty": 0.4, "agency": 0.6, "social_significance": 0.7},
    "intellectual_discovery":  {"valence": 0.8,  "goal_relevance": 0.7, "novelty": 0.9, "agency": 0.7, "social_significance": 0.3},
    "conflict_escalation":    {"valence": 0.15, "goal_relevance": 0.8, "novelty": 0.5, "agency": 0.3, "social_significance": 0.8},
    "emotional_support":      {"valence": 0.7,  "goal_relevance": 0.7, "novelty": 0.3, "agency": 0.6, "social_significance": 0.9},
    "identity_exploration":   {"valence": 0.5,  "goal_relevance": 0.9, "novelty": 0.7, "agency": 0.7, "social_significance": 0.4},
    "creative_collaboration": {"valence": 0.8,  "goal_relevance": 0.7, "novelty": 0.7, "agency": 0.7, "social_significance": 0.6},
    "boundary_setting":       {"valence": 0.3,  "goal_relevance": 0.8, "novelty": 0.4, "agency": 0.8, "social_significance": 0.7},
    "gratitude_expression":   {"valence": 0.9,  "goal_relevance": 0.6, "novelty": 0.3, "agency": 0.5, "social_significance": 0.9},
    "loss_processing":        {"valence": 0.15, "goal_relevance": 0.9, "novelty": 0.5, "agency": 0.2, "social_significance": 0.7},
    "excitement_sharing":     {"valence": 0.9,  "goal_relevance": 0.6, "novelty": 0.7, "agency": 0.6, "social_significance": 0.7},
}

# Threshold for reappraisal: if slow pass differs from fast pass by more
# than this in intensity, the slow pass overrides.
REAPPRAISAL_THRESHOLD = 0.15


def run_slow_pass(
    user_message: str,
    llm_response: str,
    slow_prototypes: dict[str, list[float]],
    fast_result: AppraisalResult,
    embedding_service: EmbeddingService,
    *,
    sensitivity: float = 0.5,
    resilience: float = 0.5,
    formative_threshold: float = 0.8,
) -> AppraisalResult:
    """Run slow-pass appraisal on the full exchange.

    Embeds the complete exchange, compares against situation prototypes.
    If the result differs significantly from the fast pass, overrides it
    (reappraisal). Otherwise, returns an enriched version of the fast result.
    """
    # Embed the full exchange
    exchange_text = f"User: {user_message}\nAssistant: {llm_response}"
    exchange_embedding = embedding_service.embed_text(exchange_text)

    # Score situation prototypes
    scores: dict[str, float] = {}
    for situation, proto_embedding in slow_prototypes.items():
        sim = cosine_similarity(exchange_embedding, proto_embedding)
        scores[situation] = max(0.0, (sim + 1.0) / 2.0)

    # Find best-matching situation
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best_situation = ranked[0][0]
    best_score = ranked[0][1]

    # Derive appraisal from situation
    total_weight = sum(s for _, s in ranked[:3])
    if total_weight == 0:
        total_weight = 1.0

    blended = {"valence": 0.0, "goal_relevance": 0.0, "novelty": 0.0,
               "agency": 0.0, "social_significance": 0.0}
    for situation, score in ranked[:3]:
        weight = score / total_weight
        appraisal_dims = SITUATION_TO_APPRAISAL.get(
            situation, list(SITUATION_TO_APPRAISAL.values())[0]
        )
        for dim, val in appraisal_dims.items():
            blended[dim] += val * weight

    slow_vector = AppraisalVector(
        goal_relevance=min(max(blended["goal_relevance"], 0.0), 1.0),
        novelty=min(max(blended["novelty"], 0.0), 1.0),
        valence=min(max(blended["valence"], 0.0), 1.0),
        agency=min(max(blended["agency"], 0.0), 1.0),
        social_significance=min(max(blended["social_significance"], 0.0), 1.0),
    )

    from emotive.layers.appraisal import calculate_half_life, calculate_intensity

    slow_intensity = calculate_intensity(slow_vector, sensitivity)
    slow_intensity = slow_intensity * min(best_score * 1.5, 1.0)

    # Reappraisal: does slow pass significantly differ from fast pass?
    intensity_diff = abs(slow_intensity - fast_result.intensity)
    valence_diff = abs(slow_vector.valence - fast_result.vector.valence)

    if intensity_diff > REAPPRAISAL_THRESHOLD or valence_diff > 0.3:
        # Slow pass overrides — reappraisal triggered
        # Use the slow pass result (more context, more accurate)
        from emotive.layers.appraisal import map_emotions

        primary, secondary = map_emotions(slow_vector)
        half_life = calculate_half_life(slow_intensity, resilience)
        is_formative = slow_intensity > formative_threshold
        decay_rate = math.log(2) / half_life if half_life > 0 else 1.0

        return AppraisalResult(
            vector=slow_vector,
            primary_emotion=primary,
            secondary_emotions=secondary,
            intensity=slow_intensity,
            half_life_minutes=half_life,
            is_formative=is_formative,
            decay_rate=decay_rate,
        )

    # No significant difference — return fast pass result (already computed)
    return fast_result
