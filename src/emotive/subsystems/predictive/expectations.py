"""Pure functions for predictive processing: expectations and prediction error.

Generates expectation embeddings from recent conversation context and computes
prediction error (surprise) when the actual next message arrives.

Brain analog: predictive coding — the brain constantly generates predictions
about upcoming sensory input and computes prediction error when reality
deviates from expectation.
"""

from __future__ import annotations

from emotive.subsystems.amygdala.fast_pass import cosine_similarity


def generate_expectation_embedding(
    response_embedding: list[float],
    conversation_embeddings: list[list[float]],
) -> list[float]:
    """Generate expectation for next message by averaging recent context.

    Simple approach: weighted average of response embedding (0.5)
    + mean of recent conversation embeddings (0.5).
    Returns a vector of the same dimensionality as the input.
    """
    dim = len(response_embedding)

    if not conversation_embeddings:
        return list(response_embedding)

    # Mean of recent conversation embeddings
    n = len(conversation_embeddings)
    context_mean = [0.0] * dim
    for emb in conversation_embeddings:
        for i in range(dim):
            context_mean[i] += emb[i]
    for i in range(dim):
        context_mean[i] /= n

    # Weighted blend: 0.5 * response + 0.5 * context mean
    result = [0.0] * dim
    for i in range(dim):
        result[i] = 0.5 * response_embedding[i] + 0.5 * context_mean[i]

    return result


def compute_prediction_error(
    expected_embedding: list[float] | None,
    actual_embedding: list[float],
) -> float:
    """Compute surprise from expected vs actual.

    If no expectation (first message): return 0.5 (neutral).
    Otherwise: 1.0 - cosine_similarity(expected, actual).
    Clamped to [0.0, 1.0].
    """
    if expected_embedding is None:
        return 0.5

    sim = cosine_similarity(expected_embedding, actual_embedding)
    error = 1.0 - sim
    return max(0.0, min(1.0, error))
