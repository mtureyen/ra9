"""Tests for connection pool safety fixes.

Covers:
- Mood subsystem uses single session in _on_episode (not 3 separate ones)
- recall_memories batches MEMORY_RECALLED events (not one per memory)
- search_by_embedding returns is_labile and primary_emotion columns
"""

from unittest.mock import MagicMock, call, patch

import pytest


class TestMoodSingleSession:
    """Mood._on_episode should use ONE session for save + history + homeostasis."""

    def test_on_episode_opens_one_session(self, app_context, event_bus):
        """The _on_episode handler should open only one DB session,
        not three (one for save, one for history, one for homeostasis).
        This was the primary cause of QueuePool exhaustion.
        """
        from emotive.subsystems.raphe import MoodSubsystem

        mood = MoodSubsystem(app_context, event_bus)

        # Count how many times session_factory is called during _on_episode
        original_factory = app_context.session_factory
        call_count = 0
        original_session = original_factory()

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return original_session

        app_context.session_factory = counting_factory

        # Reset count after MoodSubsystem init (which may call factory)
        call_count = 0

        # Trigger _on_episode directly
        mood._on_episode("episode_created", {
            "primary_emotion": "joy",
            "intensity": 0.6,
        })

        # Should open exactly 1 session for save+history (plus 1 for DB event handler)
        # The _on_episode itself should call session_factory exactly once
        # (the event bus DB handler may add 1 more for the MOOD_UPDATED event)
        assert call_count <= 2, (
            f"_on_episode opened {call_count} sessions, expected <= 2 "
            "(1 for mood DB ops + 1 for MOOD_UPDATED event handler)"
        )

        # Restore
        app_context.session_factory = original_factory


class TestRecallBatchedEvents:
    """recall_memories should publish one batched MEMORY_RECALLED event."""

    def test_recall_publishes_single_event(
        self, db_session, embedding_service, event_bus
    ):
        """Previously published one event per recalled memory,
        each opening a DB handler session. Now batched into one.
        """
        from emotive.memory.base import recall_memories, store_memory

        # Store some memories
        for text in ["hello world", "goodbye world", "test memory"]:
            store_memory(
                db_session, embedding_service,
                content=text, memory_type="episodic",
            )
        db_session.flush()

        # Track events
        events = []
        event_bus.subscribe("memory_recalled", lambda t, d: events.append(d))

        results = recall_memories(
            db_session, embedding_service,
            query="hello world",
            limit=5,
            event_bus=event_bus,
        )

        # Should be at most 1 MEMORY_RECALLED event, not N
        assert len(events) <= 1
        if events:
            assert "count" in events[0]  # batched format


class TestSearchByEmbeddingColumns:
    """search_by_embedding should return is_labile and primary_emotion."""

    def test_returns_is_labile_column(
        self, db_session, embedding_service
    ):
        """is_labile must be in search results for reconsolidation to work."""
        from emotive.db.queries.memory_queries import search_by_embedding
        from emotive.memory.base import store_memory

        store_memory(
            db_session, embedding_service,
            content="test labile check", memory_type="episodic",
        )
        db_session.flush()

        embedding = embedding_service.embed_text("test labile check")
        results = search_by_embedding(db_session, embedding, limit=5)

        assert len(results) > 0
        # The column should be present in the result dict
        assert "is_labile" in results[0]

    def test_returns_primary_emotion_column(
        self, db_session, embedding_service
    ):
        """primary_emotion must be in results for mood-congruent recall."""
        from emotive.db.queries.memory_queries import search_by_embedding
        from emotive.memory.base import store_memory

        store_memory(
            db_session, embedding_service,
            content="test emotion check", memory_type="episodic",
            primary_emotion="joy",
        )
        db_session.flush()

        embedding = embedding_service.embed_text("test emotion check")
        results = search_by_embedding(db_session, embedding, limit=5)

        assert len(results) > 0
        assert "primary_emotion" in results[0]
