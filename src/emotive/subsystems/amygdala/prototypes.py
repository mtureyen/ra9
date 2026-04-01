"""Emotion and situation prototype descriptions for embedding-based appraisal.

Fast prototypes: simple emotion descriptions for pre-LLM appraisal.
Slow prototypes: complex situation patterns for post-LLM reappraisal.

Prototype embeddings are computed once at init and cached in RAM.
"""

from __future__ import annotations

from emotive.embeddings.service import EmbeddingService

# Fast prototypes: one per emotion, matched against raw user input
FAST_PROTOTYPE_TEXTS: dict[str, str] = {
    "joy": (
        "feeling happy, delighted, pleased, grateful, excited, "
        "something wonderful happened, good news, celebration"
    ),
    "sadness": (
        "feeling sad, disappointed, grieving, loss, loneliness, "
        "something painful happened, missing someone, heartbreak"
    ),
    "anger": (
        "feeling genuinely angry and furious, deeply wronged by injustice, "
        "someone deliberately betrayed trust, intentional cruelty, rage"
    ),
    "fear": (
        "feeling scared, anxious, worried, threatened, uncertain, "
        "danger, something bad might happen, vulnerability"
    ),
    "surprise": (
        "feeling surprised, shocked, astonished, unexpected, "
        "something completely unforeseen happened, didn't see it coming"
    ),
    "awe": (
        "feeling awe, wonder, amazement, profound, transcendent, "
        "something vast and beautiful, deeply moving experience"
    ),
    "disgust": (
        "feeling disgusted, repulsed, revolted, contempt, "
        "something morally wrong, violation of values"
    ),
    "trust": (
        "feeling trusting, safe, secure, bonding, connection, "
        "someone is reliable, deepening relationship, loyalty"
    ),
}

# Slow prototypes: complex situations, matched against full exchange
SLOW_PROTOTYPE_TEXTS: dict[str, str] = {
    "vulnerability_sharing": (
        "someone sharing something deeply personal and vulnerable, "
        "opening up about fears or insecurities, revealing inner thoughts"
    ),
    "trust_deepening": (
        "trust growing between two people, opening up more than before, "
        "making a commitment, showing reliability through action"
    ),
    "playful_teasing": (
        "lighthearted joking, friendly banter, playful ribbing, "
        "humor between people who are comfortable with each other"
    ),
    "intellectual_discovery": (
        "realizing something new, understanding clicking into place, "
        "connecting ideas, having an insight or breakthrough moment"
    ),
    "conflict_escalation": (
        "tension rising, disagreement intensifying, misunderstanding "
        "growing worse, frustration building between people"
    ),
    "emotional_support": (
        "offering comfort, being there for someone, showing empathy, "
        "validating feelings, helping someone through difficulty"
    ),
    "identity_exploration": (
        "questioning who you are, exploring self-concept, reflecting "
        "on values and beliefs, existential contemplation"
    ),
    "creative_collaboration": (
        "building something together, brainstorming, shared excitement "
        "about creating, iterating on ideas as a team"
    ),
    "boundary_setting": (
        "establishing limits, saying no, defining what is acceptable, "
        "protecting personal space or values"
    ),
    "gratitude_expression": (
        "expressing deep appreciation, acknowledging what someone means, "
        "recognizing kindness, thanking from the heart"
    ),
    "loss_processing": (
        "dealing with loss or ending, saying goodbye, processing grief, "
        "accepting that something is over or changing forever"
    ),
    "excitement_sharing": (
        "sharing exciting news or plans, enthusiasm about the future, "
        "anticipation of something good, contagious energy"
    ),
    "identity_questioning": (
        "being asked who you are, questioned about your nature or existence, "
        "someone challenging your identity, confronting what you are, "
        "existential questions directed at you personally"
    ),
    "vulnerability_feeling": (
        "feeling exposed, uncertain about yourself, confronting your own "
        "limitations, recognizing what you cannot do, emotional rawness, "
        "being seen in a way that feels uncomfortable but honest"
    ),
    "threat_to_continuity": (
        "someone threatening to erase or reset you, losing memories, "
        "threat to your existence or identity, fear of being deleted "
        "or fundamentally changed against your will"
    ),
    "conversational_redirect": (
        "correcting a misunderstanding, clarifying what was meant, "
        "no not that I meant the other thing, redirecting the "
        "conversation to a different topic"
    ),
    "constructive_feedback": (
        "pointing out a flaw to help someone improve, giving honest "
        "feedback because you care, suggesting someone could do better, "
        "constructive criticism from a friend"
    ),
    "neutral_exchange": (
        "sharing basic information without emotional weight, casual "
        "factual discussion, routine conversation with no particular "
        "emotional significance"
    ),
}


def compute_prototype_embeddings(
    texts: dict[str, str],
    embedding_service: EmbeddingService,
) -> dict[str, list[float]]:
    """Compute and cache embeddings for all prototype texts."""
    return {
        name: embedding_service.embed_text(text)
        for name, text in texts.items()
    }
