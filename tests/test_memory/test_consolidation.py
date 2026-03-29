"""Tests for the consolidation engine."""

import uuid

from emotive.config.schema import EmotiveConfig
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
