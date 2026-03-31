"""Tests for Phase 1.5 foundation: event types, subsystem base, identity module."""

from emotive.runtime.event_bus import (
    APPRAISAL_COMPLETE,
    ENCODING_COMPLETE,
    EPISODE_CREATED,
    FAST_APPRAISAL_COMPLETE,
    GIST_CREATED,
    INPUT_RECEIVED,
    MEMORIES_RECALLED,
    RESPONSE_GENERATED,
    SELF_SCHEMA_REGENERATED,
    EventBus,
)
from emotive.subsystems import Subsystem


class TestNewEventTypes:
    def test_all_phase15_events_exist(self):
        assert INPUT_RECEIVED == "input_received"
        assert FAST_APPRAISAL_COMPLETE == "fast_appraisal_complete"
        assert APPRAISAL_COMPLETE == "appraisal_complete"
        assert MEMORIES_RECALLED == "memories_recalled"
        assert RESPONSE_GENERATED == "response_generated"
        assert ENCODING_COMPLETE == "encoding_complete"
        assert EPISODE_CREATED == "episode_created"
        assert GIST_CREATED == "gist_created"
        assert SELF_SCHEMA_REGENERATED == "self_schema_regenerated"

    def test_episode_created_importable_from_event_bus(self):
        """EPISODE_CREATED was moved from episodes.py to event_bus.py."""
        from emotive.layers.episodes import EPISODE_CREATED as ep_created

        assert ep_created == EPISODE_CREATED

    def test_event_bus_publishes_new_events(self):
        bus = EventBus()
        received = []
        bus.subscribe(APPRAISAL_COMPLETE, lambda t, d: received.append((t, d)))
        bus.publish(APPRAISAL_COMPLETE, {"emotion": "joy"})
        assert len(received) == 1
        assert received[0][0] == "appraisal_complete"


class TestSubsystemBase:
    def test_instantiation(self, app_context, event_bus):
        sub = Subsystem(app_context, event_bus)
        assert sub._app is app_context
        assert sub._bus is event_bus

    def test_custom_subsystem(self, app_context, event_bus):
        events = []

        class TestSub(Subsystem):
            name = "test_region"

            def _register_handlers(self):
                self._bus.subscribe("test_event", lambda t, d: events.append(d))

        sub = TestSub(app_context, event_bus)
        assert sub.name == "test_region"

        event_bus.publish("test_event", {"data": 1})
        assert len(events) == 1


class TestRecallMemoriesEmbeddingParam:
    def test_accepts_query_embedding(self, db_session, embedding_service):
        """recall_memories should accept pre-computed embedding."""
        from emotive.memory.base import recall_memories

        # Pre-compute embedding
        embedding = embedding_service.embed_text("test query")

        # Should not raise — accepts the embedding param
        results = recall_memories(
            db_session,
            embedding_service,
            query="test query",
            query_embedding=embedding,
            limit=5,
        )
        assert isinstance(results, list)

    def test_skips_embed_when_provided(self, db_session, embedding_service):
        """When query_embedding is provided, it should use that vector."""
        from unittest.mock import patch

        from emotive.memory.base import recall_memories

        embedding = embedding_service.embed_text("test query")

        with patch.object(embedding_service, "embed_text") as mock_embed:
            recall_memories(
                db_session,
                embedding_service,
                query="test query",
                query_embedding=embedding,
                limit=5,
            )
            # embed_text should NOT be called since we provided the embedding
            mock_embed.assert_not_called()


class TestIdentityModule:
    def test_load_identity_memories(self, db_session, embedding_service):
        """Identity module should return memories."""
        from emotive.memory.identity import load_identity_memories

        results = load_identity_memories(db_session)
        assert isinstance(results, list)

    def test_identity_used_by_begin_session(self):
        """begin_session should import from the shared identity module."""
        from emotive.tools.composite.begin_session import load_identity_memories

        assert callable(load_identity_memories)
