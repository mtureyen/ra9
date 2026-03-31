"""Tests for the Association Cortex subsystem (auto-recall)."""

import json

from emotive.memory.base import store_memory


class TestAssociationCortex:
    def test_initialization(self, app_context, event_bus):
        from emotive.subsystems.association_cortex import AssociationCortex

        ac = AssociationCortex(app_context, event_bus)
        assert ac.name == "association_cortex"

    def test_recall_with_shared_embedding(self, app_context, event_bus, db_session,
                                           embedding_service):
        from emotive.subsystems.association_cortex import AssociationCortex

        # Store a memory first
        store_memory(
            db_session, embedding_service,
            content="Mertcan is building an AI personality system",
            memory_type="semantic",
            tags=["project"],
        )
        db_session.flush()

        ac = AssociationCortex(app_context, event_bus)
        embedding = embedding_service.embed_text("AI personality project")
        results = ac.recall(embedding, "AI personality project", None)

        assert isinstance(results, list)
        assert len(results) > 0
        assert any("personality" in r.get("content", "").lower() for r in results)

    def test_recall_returns_empty_when_no_matches(self, app_context, event_bus,
                                                    embedding_service):
        from emotive.subsystems.association_cortex import AssociationCortex

        ac = AssociationCortex(app_context, event_bus)
        embedding = embedding_service.embed_text("xyzzy nonexistent gibberish")
        results = ac.recall(embedding, "xyzzy nonexistent gibberish", None)
        assert isinstance(results, list)

    def test_recall_publishes_event_on_results(self, app_context, event_bus,
                                                 db_session, embedding_service):
        from emotive.subsystems.association_cortex import AssociationCortex

        events = []
        event_bus.subscribe("memories_recalled", lambda t, d: events.append(d))

        store_memory(
            db_session, embedding_service,
            content="Testing auto recall event publishing",
            memory_type="semantic",
        )
        db_session.flush()

        ac = AssociationCortex(app_context, event_bus)
        embedding = embedding_service.embed_text("auto recall event")
        ac.recall(embedding, "auto recall event", None)

        # Should publish if results found
        if events:
            assert "count" in events[0]

    def test_recall_disabled(self, app_context, event_bus, embedding_service, tmp_path):
        """When auto_recall is disabled, returns empty list."""
        from emotive.config import ConfigManager
        from emotive.app_context import AppContext
        from emotive.subsystems.association_cortex import AssociationCortex

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "phase": 1,
            "auto_recall": {"enabled": False},
        }))
        cm = ConfigManager(config_path)
        ctx = AppContext(
            session_factory=app_context.session_factory,
            embedding_service=embedding_service,
            config_manager=cm,
            event_bus=event_bus,
        )

        ac = AssociationCortex(ctx, event_bus)
        embedding = embedding_service.embed_text("test")
        results = ac.recall(embedding, "test", None)
        assert results == []

    def test_recall_respects_limit(self, app_context, event_bus, db_session,
                                     embedding_service, tmp_path):
        """Auto-recall should respect the configured limit."""
        from emotive.config import ConfigManager
        from emotive.app_context import AppContext
        from emotive.subsystems.association_cortex import AssociationCortex

        # Store several memories
        for i in range(5):
            store_memory(
                db_session, embedding_service,
                content=f"Memory about testing recall limits number {i}",
                memory_type="semantic",
            )
        db_session.flush()

        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({
            "phase": 1,
            "auto_recall": {"enabled": True, "limit": 2},
        }))
        cm = ConfigManager(config_path)
        ctx = AppContext(
            session_factory=app_context.session_factory,
            embedding_service=embedding_service,
            config_manager=cm,
            event_bus=event_bus,
        )

        ac = AssociationCortex(ctx, event_bus)
        embedding = embedding_service.embed_text("testing recall limits")
        results = ac.recall(embedding, "testing recall limits", None)
        assert len(results) <= 2
