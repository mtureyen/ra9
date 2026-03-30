"""Tests for orphaned session cleanup."""

from datetime import datetime, timezone

from emotive.db.models.conversation import Conversation
from emotive.memory.session_cleanup import close_orphaned_sessions


def test_close_orphaned_sessions_closes_open_conversations(
    db_session, embedding_service, config, event_bus,
):
    # Count existing orphans first (from Ryo's live sessions)
    from sqlalchemy import select

    existing = len(list(db_session.execute(
        select(Conversation).where(Conversation.ended_at.is_(None))
    ).scalars().all()))

    # Create a new orphaned conversation
    conv = Conversation(metadata_={})
    db_session.add(conv)
    db_session.flush()

    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus,
    )

    # Should have cleaned at least our new orphan
    assert cleaned >= existing + 1
    db_session.refresh(conv)
    assert conv.ended_at is not None


def test_close_orphaned_sessions_skips_closed(
    db_session, embedding_service, config, event_bus,
):
    from sqlalchemy import select

    # Close all existing orphans first
    orphans = list(db_session.execute(
        select(Conversation).where(Conversation.ended_at.is_(None))
    ).scalars().all())
    for o in orphans:
        o.ended_at = datetime.now(timezone.utc)
    db_session.flush()

    # Create an already-closed conversation
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
    from sqlalchemy import select

    # Close all existing orphans first
    orphans = list(db_session.execute(
        select(Conversation).where(Conversation.ended_at.is_(None))
    ).scalars().all())
    for o in orphans:
        o.ended_at = datetime.now(timezone.utc)
    db_session.flush()

    # Create 5 new orphans
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
    from sqlalchemy import select

    # Close all existing orphans first
    orphans = list(db_session.execute(
        select(Conversation).where(Conversation.ended_at.is_(None))
    ).scalars().all())
    for o in orphans:
        o.ended_at = datetime.now(timezone.utc)
    db_session.flush()

    cleaned = close_orphaned_sessions(
        db_session, embedding_service, config, event_bus=event_bus,
    )
    assert cleaned == 0
