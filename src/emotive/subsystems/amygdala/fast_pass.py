"""Fast-pass appraisal: pre-LLM, embedding-based emotion detection.

Runs on the raw user input BEFORE the LLM sees it. Compares the input
embedding against pre-computed emotion prototype embeddings using cosine
similarity. ~5ms. No LLM call.

Brain analog: amygdala via thalamic low road (50-150ms, subcortical).
"""

from __future__ import annotations

import math

from emotive.layers.appraisal import AppraisalResult, AppraisalVector


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# Mapping from emotion names to approximate appraisal dimensions.
# Used to derive an AppraisalVector from emotion similarity scores.
EMOTION_TO_APPRAISAL: dict[str, dict[str, float]] = {
    "joy":      {"valence": 0.85, "goal_relevance": 0.7, "novelty": 0.4, "agency": 0.6, "social_significance": 0.5},
    "sadness":  {"valence": 0.15, "goal_relevance": 0.7, "novelty": 0.3, "agency": 0.2, "social_significance": 0.5},
    "anger":    {"valence": 0.15, "goal_relevance": 0.7, "novelty": 0.3, "agency": 0.2, "social_significance": 0.6},
    "fear":     {"valence": 0.15, "goal_relevance": 0.6, "novelty": 0.7, "agency": 0.2, "social_significance": 0.3},
    "surprise": {"valence": 0.5,  "goal_relevance": 0.5, "novelty": 0.9, "agency": 0.3, "social_significance": 0.3},
    "awe":      {"valence": 0.8,  "goal_relevance": 0.6, "novelty": 0.8, "agency": 0.3, "social_significance": 0.5},
    "disgust":  {"valence": 0.1,  "goal_relevance": 0.5, "novelty": 0.3, "agency": 0.2, "social_significance": 0.6},
    "trust":    {"valence": 0.8,  "goal_relevance": 0.6, "novelty": 0.2, "agency": 0.5, "social_significance": 0.8},
}


def run_fast_pass(
    input_embedding: list[float],
    prototypes: dict[str, list[float]],
    *,
    sensitivity: float = 0.5,
    resilience: float = 0.5,
    formative_threshold: float = 0.8,
) -> AppraisalResult:
    """Run fast-pass appraisal on a pre-computed input embedding.

    Compares input against emotion prototypes via cosine similarity.
    Derives an AppraisalVector from the weighted emotion scores.
    Returns a full AppraisalResult.
    """
    # Score each emotion prototype
    scores: dict[str, float] = {}
    for emotion, proto_embedding in prototypes.items():
        sim = cosine_similarity(input_embedding, proto_embedding)
        # Normalize similarity from [-1, 1] to [0, 1]
        scores[emotion] = max(0.0, (sim + 1.0) / 2.0)

    # Primary emotion = highest score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    primary_emotion = ranked[0][0]
    primary_score = ranked[0][1]

    # Secondary emotions (next 1-2 if score > 0.55)
    secondary = [e for e, s in ranked[1:3] if s > 0.55]

    # Derive AppraisalVector as weighted blend of top emotion appraisals
    # Weight by similarity score, normalize
    total_weight = sum(s for _, s in ranked[:3])
    if total_weight == 0:
        total_weight = 1.0

    blended = {"valence": 0.0, "goal_relevance": 0.0, "novelty": 0.0,
               "agency": 0.0, "social_significance": 0.0}
    for emotion, score in ranked[:3]:
        weight = score / total_weight
        appraisal_dims = EMOTION_TO_APPRAISAL.get(emotion, EMOTION_TO_APPRAISAL["surprise"])
        for dim, val in appraisal_dims.items():
            blended[dim] += val * weight

    vector = AppraisalVector(
        goal_relevance=min(max(blended["goal_relevance"], 0.0), 1.0),
        novelty=min(max(blended["novelty"], 0.0), 1.0),
        valence=min(max(blended["valence"], 0.0), 1.0),
        agency=min(max(blended["agency"], 0.0), 1.0),
        social_significance=min(max(blended["social_significance"], 0.0), 1.0),
    )

    # Calculate intensity (same formula as layers/appraisal.py)
    from emotive.layers.appraisal import calculate_half_life, calculate_intensity

    intensity = calculate_intensity(vector, sensitivity)

    # Scale intensity by how confident the match is
    # (high similarity to best prototype = more confident)
    intensity = intensity * min(primary_score * 1.5, 1.0)

    half_life = calculate_half_life(intensity, resilience)
    is_formative = intensity > formative_threshold
    decay_rate = math.log(2) / half_life if half_life > 0 else 1.0

    return AppraisalResult(
        vector=vector,
        primary_emotion=primary_emotion,
        secondary_emotions=secondary,
        intensity=intensity,
        half_life_minutes=half_life,
        is_formative=is_formative,
        decay_rate=decay_rate,
    )
