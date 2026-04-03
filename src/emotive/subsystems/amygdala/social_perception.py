"""Social perception: reads the user's emotional state from their message.

A second output channel from the amygdala fast pass. Compares the user's
input embedding against social perception prototypes to infer what the
user might be feeling or intending.

Brain analog: social cognition / theory of mind circuits in the amygdala
and superior temporal sulcus.
"""

from __future__ import annotations

from emotive.embeddings.service import EmbeddingService
from emotive.subsystems.amygdala.fast_pass import cosine_similarity

# Social perception prototypes: descriptions of what the USER might be feeling.
USER_STATE_PROTOTYPE_TEXTS: dict[str, str] = {
    "curious": (
        "someone who is genuinely curious and wants to understand, "
        "asking real questions out of interest"
    ),
    "testing": (
        "someone who is deliberately probing, checking, examining, "
        "running a test or experiment"
    ),
    "upset": (
        "someone who is frustrated, hurt, angry, or emotionally "
        "distressed about something"
    ),
    "playful": (
        "someone who is being light, joking, having fun, "
        "not taking things too seriously"
    ),
    "vulnerable": (
        "someone who is opening up about something personal, "
        "sharing something they normally wouldn't"
    ),
    "enthusiastic": (
        "someone who is excited, positive, energized, "
        "genuinely happy about something"
    ),
    "withdrawn": (
        "someone who is pulling back, giving short answers, "
        "disengaging, losing interest"
    ),
    "confrontational": (
        "someone who is challenging, pushing back, arguing, "
        "being deliberately adversarial"
    ),
}


def compute_social_perception_prototypes(
    prototype_texts: dict[str, str],
    embedding_service: EmbeddingService,
) -> dict[str, list[float]]:
    """Compute and cache embeddings for social perception prototype texts."""
    return {
        name: embedding_service.embed_text(text)
        for name, text in prototype_texts.items()
    }


def run_social_perception(
    input_embedding: list[float],
    prototypes: dict[str, list[float]],
) -> tuple[str | None, float]:
    """Infer the user's emotional state from their message embedding.

    Compares input against social perception prototypes via cosine similarity.

    Returns:
        (user_state, confidence) where user_state is None if confidence
        is below the threshold (can't read the user clearly).
    """
    scores: dict[str, float] = {}
    for state, proto_embedding in prototypes.items():
        sim = cosine_similarity(input_embedding, proto_embedding)
        # Normalize similarity from [-1, 1] to [0, 1]
        scores[state] = (sim + 1.0) / 2.0

    if not scores:
        return (None, 0.0)

    # Find best match
    best_state = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_state]

    # Confidence gate: if best match < 0.50, can't read the user clearly
    if best_score < 0.50:
        return (None, 0.0)

    return (best_state, best_score)
