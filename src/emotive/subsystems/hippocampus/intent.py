"""Conscious intent detection: catches "I want to remember this" in LLM output.

When the LLM expresses intent to remember something, the post-processor
detects it and triggers enhanced encoding — more retrieval cues, higher
significance, stronger decay protection.

Brain analog: dlPFC attentional amplification + LIFG elaborative encoding.
"""

from __future__ import annotations

import re
import uuid

from sqlalchemy.orm import Session

from emotive.embeddings.service import EmbeddingService
from emotive.logging import get_logger
from emotive.memory.base import store_memory
from emotive.runtime.event_bus import ENCODING_COMPLETE, EventBus

logger = get_logger("hippocampus.intent")

# Patterns that indicate conscious encoding intent
INTENT_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"i want to remember",
        r"i need to keep this in mind",
        r"this is important to me",
        r"i should remember",
        r"i'll remember",
        r"don'?t let me forget",
        r"note to self",
        r"i need to remember",
        r"let me hold on to",
        r"i want to keep this",
        r"this matters to me",
        r"i won'?t forget",
    ]
]


def detect_encoding_intent(text: str) -> bool:
    """Scan text for conscious encoding intent patterns."""
    return any(p.search(text) for p in INTENT_PATTERNS)


def enhanced_encode(
    session: Session,
    embedding_service: EmbeddingService,
    content: str,
    *,
    conversation_id: uuid.UUID | None = None,
    event_bus: EventBus | None = None,
) -> None:
    """Store with enhanced encoding: higher significance, lower decay.

    This is the conscious encoding path — triggered by the LLM expressing
    intent to remember. Same store_memory() function, just with stronger
    parameters. No origin tag — indistinguishable from unconscious memories.
    """
    memory = store_memory(
        session,
        embedding_service,
        content=content[:500],
        memory_type="episodic",
        conversation_id=conversation_id,
        metadata={"significance": 0.85},
        decay_protection=0.5,
        tags=["conscious_intent"],
        event_bus=event_bus,
    )

    if event_bus:
        event_bus.publish(
            ENCODING_COMPLETE,
            {
                "source": "conscious_intent",
                "content": content[:200],
            },
            memory_id=memory.id,
            conversation_id=conversation_id,
        )

    logger.info("Enhanced encoding (conscious intent): memory %s", memory.id)
