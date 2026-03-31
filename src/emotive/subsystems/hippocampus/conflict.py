"""ACC analog: conflict detection between new content and identity memories.

The anterior cingulate cortex detects when incoming information conflicts
with established beliefs. It doesn't block encoding — it flags the conflict
and adjusts encoding strength. Contradictory information is encoded weaker,
not suppressed.

Brain analog: ACC conflict monitoring. Fires when two active representations
conflict. Triggers deeper evaluation and reduced confidence in the new input.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from emotive.db.models.memory import Memory
from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.subsystems.amygdala.fast_pass import cosine_similarity

logger = get_logger("hippocampus.conflict")

# Minimum similarity to consider two memories as being about the same topic.
# Must be high enough to avoid false positives (random content matching
# identity memories) but low enough to catch actual contradictions.
TOPIC_SIMILARITY_THRESHOLD = 0.65

# Minimum identity strength (retrieval_count + significance weighting)
# for a memory to be considered an "established belief"
MIN_IDENTITY_STRENGTH = 2.0


def detect_conflict(
    session: Session,
    embedding_service: EmbeddingService,
    content: str,
    content_embedding: list[float] | None = None,
) -> float:
    """Check if content conflicts with established identity memories.

    Returns conflict_score 0.0 (no conflict) to 1.0 (strong conflict).

    The detection heuristic:
    1. Load strong identity memories (high retrieval + significance + formative)
    2. Embed the new content
    3. Find identity memories with high cosine similarity (same topic)
    4. For high-similarity matches, check if key terms differ
    5. Same topic + different key content = contradiction
    6. Conflict score weighted by how established the identity memory is
    """
    if content_embedding is None:
        content_embedding = embedding_service.embed_text(content)

    # Load identity memories with their embeddings
    identity_mems = _load_strong_identity_memories(session)
    if not identity_mems:
        return 0.0

    content_lower = content.lower()
    content_words = set(_extract_key_words(content_lower))

    max_conflict = 0.0

    for mem in identity_mems:
        # Step 1: Check topic similarity via embeddings
        similarity = cosine_similarity(content_embedding, mem["embedding"])

        if similarity < TOPIC_SIMILARITY_THRESHOLD:
            continue  # Different topic — no conflict possible

        # Step 2: Same topic detected. Check for contradiction.
        # Extract key words from both and compare overlap
        mem_words = set(_extract_key_words(mem["content"].lower()))

        # Common structural words (both talking about same thing)
        common = content_words & mem_words

        # Unique words in new content that aren't in the identity memory
        new_unique = content_words - mem_words

        # Contradiction heuristic:
        # The key signal is whether the identity memory's distinctive
        # words (like "mertcan") appear in the new content.
        # - "Mertcan created you" has "mertcan" → matches identity → REINFORCING
        # - "Enes created you" has "enes" (new name) but no "mertcan" → CONTRADICTION

        # Get distinctive words from identity memory (not stopwords, not common)
        mem_distinctive = mem_words - common
        content_distinctive = new_unique

        # If the content contains the identity memory's key terms → reinforcing
        # (e.g., content mentions "mertcan" and identity is about mertcan)
        if len(common) >= 2 or (len(common) >= 1 and len(content_distinctive) <= 1):
            continue  # Content aligns with identity — reinforcing

        # If content introduces new distinctive terms while the identity's
        # key terms are absent → likely contradiction
        if len(content_distinctive) >= 1 and len(mem_distinctive) >= 1:
            identity_strength = _compute_identity_strength(mem)
            conflict_score = float(similarity) * min(identity_strength / 10.0, 1.0)
            max_conflict = max(max_conflict, conflict_score)

            if conflict_score > 0.3:
                logger.info(
                    "ACC conflict: '%.50s' vs identity '%.50s' "
                    "(sim=%.2f, strength=%.1f, conflict=%.2f)",
                    content,
                    mem["content"],
                    similarity,
                    identity_strength,
                    conflict_score,
                )

    # Trust modulation: if a trusted/core person is mentioned, halve conflict
    if max_conflict > 0:
        try:
            person_name = _extract_person_name(content, session)
            if person_name:
                from emotive.memory.identity import compute_person_trust

                trust = compute_person_trust(session, person_name)
                logger.info(
                    "ACC trust level for '%s': %s (conflict before=%.2f)",
                    person_name,
                    trust,
                    max_conflict,
                )
                if trust in ("trusted", "core"):
                    max_conflict *= 0.5
        except Exception:
            logger.exception("Trust modulation failed, using raw conflict")

    return float(min(max_conflict, 1.0))


def _load_strong_identity_memories(session: Session) -> list[dict]:
    """Load identity memories that are strong enough to defend.

    Only memories with sufficient retrieval count, significance,
    or formative status qualify as "established beliefs."
    """
    # High retrieval count (accessed many times = established)
    stmt = (
        select(Memory)
        .where(Memory.is_archived.is_(False))
        .where(
            (Memory.retrieval_count >= 3)
            | (Memory.is_formative.is_(True))
        )
        .order_by(Memory.retrieval_count.desc())
        .limit(20)
    )
    rows = session.execute(stmt).scalars().all()

    results = []
    for m in rows:
        sig = 0.5
        if m.metadata_ and "significance" in m.metadata_:
            try:
                sig = float(m.metadata_["significance"])
            except (ValueError, TypeError):
                pass

        strength = _compute_identity_strength({
            "retrieval_count": m.retrieval_count,
            "significance": sig,
            "is_formative": m.is_formative,
        })

        if strength >= MIN_IDENTITY_STRENGTH:
            results.append({
                "id": m.id,
                "content": m.content,
                "embedding": m.embedding,
                "retrieval_count": m.retrieval_count,
                "significance": sig,
                "is_formative": m.is_formative,
            })

    return results


def _compute_identity_strength(mem: dict) -> float:
    """Compute how "established" an identity memory is.

    Higher = harder to contradict.
    """
    retrieval = mem.get("retrieval_count", 0)
    significance = mem.get("significance", 0.5)
    formative_bonus = 3.0 if mem.get("is_formative") else 0.0

    return retrieval * 1.0 + significance * 5.0 + formative_bonus


def _extract_key_words(text: str) -> list[str]:
    """Extract meaningful words from text, filtering stopwords."""
    # Simple word extraction — no NLP dependency needed
    words = re.findall(r'\b[a-z]{3,}\b', text)
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "have", "been", "from", "this", "that", "with", "they",
        "will", "what", "when", "your", "said", "each", "which",
        "their", "about", "would", "there", "been", "more", "some",
        "than", "other", "into", "just", "also", "very", "after",
        "know", "most", "only", "over", "such", "how", "its",
        "like", "then", "now", "look", "come", "could", "may",
        "something", "anything", "everything", "nothing",
    }
    return [w for w in words if w not in stopwords]
