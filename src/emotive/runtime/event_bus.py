"""Synchronous pub/sub event bus. Bridges internal events to event_log table."""

from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Callable

from emotive.logging import get_logger, log_event

logger = get_logger("event_bus")

# Event type constants
MEMORY_STORED = "memory_stored"
MEMORY_RECALLED = "memory_recalled"
MEMORY_DECAYED = "memory_decayed"
MEMORY_ARCHIVED = "memory_archived"
MEMORY_LINKED = "memory_linked"
SESSION_STARTED = "session_started"
SESSION_ENDED = "session_ended"
CONSOLIDATION_STARTED = "consolidation_started"
CONSOLIDATION_COMPLETED = "consolidation_completed"
CONFIG_CHANGED = "config_changed"
WORKING_MEMORY_EVICTED = "working_memory_evicted"

# Phase 1.5: cognitive pipeline events
INPUT_RECEIVED = "input_received"
FAST_APPRAISAL_COMPLETE = "fast_appraisal_complete"
APPRAISAL_COMPLETE = "appraisal_complete"
MEMORIES_RECALLED = "memories_recalled"
RESPONSE_GENERATED = "response_generated"
ENCODING_COMPLETE = "encoding_complete"
EPISODE_CREATED = "episode_created"
GIST_CREATED = "gist_created"
SELF_SCHEMA_REGENERATED = "self_schema_regenerated"
MOOD_UPDATED = "mood_updated"

Handler = Callable[[str, dict[str, Any]], None]


class EventBus:
    """Simple synchronous event bus with subscriber handlers."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = defaultdict(list)
        self._global_handlers: list[Handler] = []

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """Subscribe a handler to a specific event type."""
        self._handlers[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Subscribe a handler to all events."""
        self._global_handlers.append(handler)

    def publish(
        self,
        event_type: str,
        data: dict[str, Any] | None = None,
        *,
        memory_id: uuid.UUID | None = None,
        conversation_id: uuid.UUID | None = None,
        episode_id: uuid.UUID | None = None,
        consolidation_id: int | None = None,
    ) -> None:
        """Publish an event to all subscribed handlers."""
        payload = data or {}

        # Attach reference IDs to payload for handlers that need them
        refs = {}
        if memory_id is not None:
            refs["memory_id"] = str(memory_id)
        if conversation_id is not None:
            refs["conversation_id"] = str(conversation_id)
        if episode_id is not None:
            refs["episode_id"] = str(episode_id)
        if consolidation_id is not None:
            refs["consolidation_id"] = consolidation_id
        if refs:
            payload = {**payload, "_refs": refs}

        # Structured log
        log_event(logger, event_type, payload)

        # Dispatch to type-specific handlers
        for handler in self._handlers.get(event_type, []):
            handler(event_type, payload)

        # Dispatch to global handlers
        for handler in self._global_handlers:
            handler(event_type, payload)


def create_db_handler(session_factory: Callable) -> Handler:
    """Create a handler that writes events to the event_log table."""
    from emotive.db.models.event_log import EventLog

    def handler(event_type: str, payload: dict[str, Any]) -> None:
        refs = payload.get("_refs", {})
        session = session_factory()
        try:
            entry = EventLog(
                event_type=event_type,
                event_data=payload,
                memory_id=uuid.UUID(refs["memory_id"]) if "memory_id" in refs else None,
                conversation_id=(
                    uuid.UUID(refs["conversation_id"]) if "conversation_id" in refs else None
                ),
                episode_id=(
                    uuid.UUID(refs["episode_id"]) if "episode_id" in refs else None
                ),
                consolidation_id=refs.get("consolidation_id"),
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

    return handler
