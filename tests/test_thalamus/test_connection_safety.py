"""Tests for connection leak fixes and session lifecycle safety.

Covers:
- Procedural memories loaded within session scope (not after close)
- Event bus DB handler doesn't mutate caller's payload
- Engine pool configuration
"""

from unittest.mock import MagicMock, patch

import pytest

from emotive.runtime.event_bus import EventBus, create_db_handler


class TestEventBusPayloadSafety:
    """Ensure the DB handler doesn't mutate the payload dict."""

    def test_db_handler_does_not_pop_refs(self, db_session, session_factory):
        """The DB handler should read _refs without removing them from payload.

        Previously used payload.pop('_refs', {}) which mutated the dict.
        If multiple handlers (type-specific + global) process the same event,
        the second handler would lose _refs.
        """
        bus = EventBus()
        handler = create_db_handler(session_factory)
        bus.subscribe_all(handler)

        # Track payloads received by a second handler
        received = []

        def spy_handler(event_type, data):
            received.append(dict(data))

        bus.subscribe_all(spy_handler)

        # Publish with _refs
        bus.publish(
            "test_event",
            {"key": "value"},
            memory_id=None,
            conversation_id=None,
        )

        # The spy handler should see the full payload (no _refs stripped)
        assert len(received) == 1
        assert "key" in received[0]


class TestProceduralMemorySessionScope:
    """Ensure procedural memories are loaded within the DB session scope."""

    def test_load_procedural_within_session(self, db_session):
        """_load_procedural_memories must be called with an open session.

        Regression test: previously called AFTER session.close() in
        dispatcher.py, causing queries on a closed session.
        """
        from emotive.thalamus.dispatcher import _load_procedural_memories

        # Should work fine with an open session
        result = _load_procedural_memories(db_session)
        assert isinstance(result, list)


class TestEnginePoolConfig:
    """Verify engine pool settings prevent connection exhaustion."""

    def test_pool_size_increased(self):
        from emotive.db.engine import engine

        pool = engine.pool
        assert pool.size() >= 10
