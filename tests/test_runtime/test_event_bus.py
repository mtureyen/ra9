"""Tests for the event bus."""

import uuid

from emotive.runtime.event_bus import (
    MEMORY_STORED,
    SESSION_STARTED,
    EventBus,
    create_db_handler,
)


def test_subscribe_and_publish_delivers_to_handler():
    bus = EventBus()
    received = []
    bus.subscribe(MEMORY_STORED, lambda t, d: received.append((t, d)))
    bus.publish(MEMORY_STORED, {"content": "test"})
    assert len(received) == 1
    assert received[0][0] == MEMORY_STORED


def test_subscribe_multiple_handlers():
    bus = EventBus()
    calls = []
    bus.subscribe(MEMORY_STORED, lambda t, d: calls.append("a"))
    bus.subscribe(MEMORY_STORED, lambda t, d: calls.append("b"))
    bus.publish(MEMORY_STORED)
    assert calls == ["a", "b"]


def test_subscribe_different_events_no_crosstalk():
    bus = EventBus()
    received = []
    bus.subscribe(MEMORY_STORED, lambda t, d: received.append(t))
    bus.publish(SESSION_STARTED)
    assert len(received) == 0


def test_subscribe_all_receives_every_event():
    bus = EventBus()
    received = []
    bus.subscribe_all(lambda t, d: received.append(t))
    bus.publish(MEMORY_STORED)
    bus.publish(SESSION_STARTED)
    assert len(received) == 2


def test_publish_includes_refs_in_payload():
    bus = EventBus()
    received = []
    bus.subscribe(MEMORY_STORED, lambda t, d: received.append(d))
    mid = uuid.uuid4()
    bus.publish(MEMORY_STORED, {"x": 1}, memory_id=mid)
    assert received[0]["_refs"]["memory_id"] == str(mid)


def test_publish_without_refs_has_no_refs_key():
    bus = EventBus()
    received = []
    bus.subscribe(MEMORY_STORED, lambda t, d: received.append(d))
    bus.publish(MEMORY_STORED, {"x": 1})
    assert "_refs" not in received[0]


def test_create_db_handler_writes_event_log_row(db_session, session_factory):
    from emotive.db.models.event_log import EventLog

    handler = create_db_handler(session_factory)
    handler(MEMORY_STORED, {"content": "db_handler_unique_test_xyz"})
    row = (
        db_session.query(EventLog)
        .filter(EventLog.event_type == MEMORY_STORED)
        .order_by(EventLog.recorded_at.desc())
        .first()
    )
    assert row is not None
    assert row.event_data["content"] == "db_handler_unique_test_xyz"
