"""Tests for the Default Mode Network subsystem (self-schema generation)."""

from emotive.memory.base import store_memory
from emotive.subsystems.dmn.schema import SelfSchema, regenerate_schema


class TestSelfSchema:
    def test_default_empty(self):
        schema = SelfSchema()
        assert schema.traits == {}
        assert schema.core_facts == []
        assert schema.active_values == []
        assert schema.person_context == {}
        assert schema.generated_at is not None


class TestRegenerateSchema:
    def test_empty_database(self, db_session):
        schema = regenerate_schema(db_session)
        assert isinstance(schema, SelfSchema)
        assert isinstance(schema.traits, dict)
        assert isinstance(schema.core_facts, list)

    def test_with_memories(self, db_session, embedding_service):
        # Store some memories with tags
        for i in range(3):
            store_memory(
                db_session, embedding_service,
                content=f"I enjoy exploring new ideas, thought {i}",
                memory_type="semantic",
                tags=["curious", "intellectual"],
                metadata={"significance": 0.8},
            )
        store_memory(
            db_session, embedding_service,
            content="My name is Ryo",
            memory_type="semantic",
            metadata={"significance": 0.95},
        )
        db_session.flush()

        schema = regenerate_schema(db_session)
        # Should have traits from tag frequencies
        assert len(schema.traits) > 0

    def test_core_facts_from_identity(self, db_session, embedding_service):
        store_memory(
            db_session, embedding_service,
            content="My name is Ryo",
            memory_type="semantic",
            metadata={"significance": 0.95},
        )
        db_session.flush()

        schema = regenerate_schema(db_session, max_core_facts=5)
        # core_facts depend on load_identity_memories finding high-significance memories

    def test_max_limits(self, db_session, embedding_service):
        for i in range(20):
            store_memory(
                db_session, embedding_service,
                content=f"Memory {i} about topics",
                memory_type="semantic",
                tags=[f"tag_{i % 5}"],
            )
        db_session.flush()

        schema = regenerate_schema(db_session, max_traits=3)
        assert len(schema.traits) <= 3


class TestDMNSubsystem:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.dmn import DefaultModeNetwork

        dmn = DefaultModeNetwork(app_context, event_bus)
        assert dmn.name == "dmn"
        assert dmn.current is None

    def test_regenerate(self, app_context, event_bus):
        from emotive.subsystems.dmn import DefaultModeNetwork

        dmn = DefaultModeNetwork(app_context, event_bus)
        schema = dmn.regenerate()
        assert isinstance(schema, SelfSchema)
        assert dmn.current is schema

    def test_publishes_event_on_regeneration(self, app_context, event_bus):
        from emotive.subsystems.dmn import DefaultModeNetwork

        events = []
        event_bus.subscribe(
            "self_schema_regenerated", lambda t, d: events.append(d)
        )

        dmn = DefaultModeNetwork(app_context, event_bus)
        dmn.regenerate()

        assert len(events) == 1
        assert "traits_count" in events[0]

    def test_auto_regenerate_on_consolidation(self, app_context, event_bus):
        from emotive.subsystems.dmn import DefaultModeNetwork

        dmn = DefaultModeNetwork(app_context, event_bus)
        assert dmn.current is None

        # Simulate consolidation completed event
        event_bus.publish("consolidation_completed", {})

        # DMN should have regenerated
        assert dmn.current is not None
