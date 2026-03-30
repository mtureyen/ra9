"""Tests for orphaned session cleanup."""

import uuid
from datetime import datetime, timezone

from emotive.db.models.conversation import Conversation
from emotive.memory.session_cleanup import close_orphaned_sessions


def test_close_orphaned_sessions_closes_open_conversations(
    db_session, embedding_service, config, event_bus,
):
    # Create an orphaned conversation (no ended_at)
    conv = Conversation(metadata_={})
    db_session.add(conv)
    db_session.flush()
    assert conv.ended_at is None

    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus,
    )
    db_session.flush()

    assert cleaned == 1
    db_session.refresh(conv)
    assert conv.ended_at is not None


def test_close_orphaned_sessions_skips_closed(
    db_session, embedding_service, config, event_bus,
):
    # Already closed conversation
    conv = Conversation(ended_at=datetime.now(timezone.utc), metadata_={})
    db_session.add(conv)
    db_session.flush()

    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus,
    )
    assert cleaned == 0


def test_close_orphaned_sessions_respects_limit(
    db_session, embedding_service, config, event_bus,
):
    # Create 5 orphans
    for _ in range(5):
        db_session.add(Conversation(metadata_={}))
    db_session.flush()

    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus, limit=3,
    )
    assert cleaned == 3


def test_close_orphaned_sessions_none_returns_zero(
    db_session, embedding_service, config, event_bus,
):
    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus,
    )
    assert cleaned == 0
