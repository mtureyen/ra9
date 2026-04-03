"""Discovery detection for self-output appraisal.

Detects when the LLM introduces concepts not present in recalled
memories -- a form of creative emergence.

Brain analog: novelty detection in hippocampus -- "this is new."
"""

from __future__ import annotations

from emotive.subsystems.amygdala.fast_pass import cosine_similarity


def detect_discovery(
    response_embedding: list[float],
    recalled_embeddings: list[list[float]],
) -> bool:
    """Did the LLM introduce concepts not in recalled memories?

    Compares response embedding against all recalled memory embeddings.
    If the response is far from all recalled memories, it's a discovery.

    Args:
        response_embedding: Embedding of the generated response.
        recalled_embeddings: Embeddings of recalled memories.

    Returns:
        True if the response contains novel concepts (discovery).
    """
    if not recalled_embeddings:
        return True  # Everything is new if nothing was recalled

    max_sim = max(
        cosine_similarity(response_embedding, mem_emb)
        for mem_emb in recalled_embeddings
    )
    return max_sim < 0.6  # Response is far from all recalled memories = discovery
