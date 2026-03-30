"""Tests for the consolidation engine."""


from emotive.db.models.consolidation import ConsolidationLog
from emotive.db.models.memory import Memory
from emotive.memory.consolidation import run_consolidation
from emotive.memory.episodic import store_episodic
from emotive.runtime.event_bus import CONSOLIDATION_COMPLETED, CONSOLIDATION_STARTED
from emotive.runtime.working_memory import WorkingMemory, WorkingMemoryItem


def test_run_consolidation_creates_log_entry(db_session, embedding_service, config, event_bus):
    result = run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    db_session.flush()
    log = db_session.query(ConsolidationLog).filter(
        ConsolidationLog.id == result["consolidation_id"]
    ).one()
    assert log.trigger_type == "conversation_end"


def test_run_consolidation_promotes_significant_items(
    db_session, embedding_service, config, event_bus,
):
    wm = WorkingMemory(capacity=10)
    wm.add(WorkingMemoryItem(content="important discussion about trust", significance=0.8))
    wm.add(WorkingMemoryItem(content="meaningful insight about growth", significance=0.6))

    result = run_consolidation(
        db_session, embedding_service, config, working_memory=wm, event_bus=event_bus,
    )
    assert result["promotion"]["working_to_episodic"] == 2


def test_run_consolidation_skips_low_significance(
    db_session, embedding_service, config, event_bus,
):
    wm = WorkingMemory(capacity=10)
    wm.add(WorkingMemoryItem(content="weather chat", significance=0.1))
    wm.add(WorkingMemoryItem(content="small talk", significance=0.2))

    result = run_consolidation(
        db_session, embedding_service, config, working_memory=wm, event_bus=event_bus,
    )
    assert result["promotion"]["working_to_episodic"] == 0


def test_run_consolidation_no_working_memory(db_session, embedding_service, config, event_bus):
    result = run_consolidation(
        db_session, embedding_service, config, working_memory=None, event_bus=event_bus,
    )
    assert result["promotion"]["working_to_episodic"] == 0


def test_run_consolidation_applies_decay(db_session, embedding_service, config, event_bus):
    store_episodic(db_session, embedding_service, content="old memory for decay")
    db_session.flush()

    result = run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    assert result["decay"]["memories_decayed"] >= 1


def test_run_consolidation_publishes_started_and_completed(
    db_session, embedding_service, config, event_bus,
):
    received = []
    event_bus.subscribe(CONSOLIDATION_STARTED, lambda t, d: received.append(t))
    event_bus.subscribe(CONSOLIDATION_COMPLETED, lambda t, d: received.append(t))

    run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    assert CONSOLIDATION_STARTED in received
    assert CONSOLIDATION_COMPLETED in received


def test_run_consolidation_report_structure(db_session, embedding_service, config, event_bus):
    result = run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    assert "consolidation_id" in result
    assert "duration_ms" in result
    assert "promotion" in result
    assert "extraction" in result
    assert "linking" in result
    assert "decay" in result


def test_run_consolidation_report_includes_concept_hubs(
    db_session, embedding_service, config, event_bus,
):
    result = run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    assert "concept_hubs_created" in result["extraction"]


def test_run_consolidation_creates_concept_hubs(
    db_session, embedding_service, config, event_bus,
):
    """When 5+ memories share a tag, consolidation creates a concept hub."""
    from emotive.memory.base import store_memory

    for i in range(6):
        store_memory(
            db_session, embedding_service,
            content=f"Cooking recipe variation number {i} with pasta",
            memory_type="episodic",
            tags=["cooking"],
        )
    db_session.flush()

    result = run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    db_session.flush()

    # Check hubs were created
    assert result["extraction"]["concept_hubs_created"] > 0
    hubs = db_session.query(Memory).filter(
        Memory.memory_type == "semantic",
        Memory.tags.contains(["concept_hub"]),
    ).all()
    assert len(hubs) > 0
    # At least one hub should be a concept hub with proper metadata
    hub_tags = [h.metadata_.get("hub_tag") for h in hubs]
    assert "cooking" in hub_tags


def test_run_consolidation_links_unlinked_memories(
    db_session, embedding_service, config, event_bus,
):
    """Consolidation should link previously isolated memories."""
    from emotive.db.models.memory import MemoryLink
    from emotive.memory.base import store_memory

    # Store two similar memories without conversation_id (no temporal links)
    m1 = store_memory(
        db_session, embedding_service,
        content="Hiking in the Swiss Alps in summer", memory_type="semantic",
    )
    m2 = store_memory(
        db_session, embedding_service,
        content="Trekking through the Alps in July", memory_type="semantic",
    )
    db_session.flush()

    run_consolidation(db_session, embedding_service, config, event_bus=event_bus)
    db_session.flush()

    # They should be linked now (either from instant link or consolidation)
    links = db_session.query(MemoryLink).filter(
        ((MemoryLink.source_memory_id == m1.id) & (MemoryLink.target_memory_id == m2.id))
        | ((MemoryLink.source_memory_id == m2.id) & (MemoryLink.target_memory_id == m1.id))
    ).all()
    assert len(links) >= 1
