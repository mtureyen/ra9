"""Tests for procedural memory."""

from emotive.memory.procedural import PROCEDURAL_DECAY_RATE, store_procedural


def test_store_procedural_sets_type(db_session, embedding_service):
    mem = store_procedural(db_session, embedding_service, content="how to apologize")
    assert mem.memory_type == "procedural"


def test_store_procedural_sets_decay_rate(db_session, embedding_service):
    mem = store_procedural(db_session, embedding_service, content="decay test")
    assert mem.decay_rate == PROCEDURAL_DECAY_RATE


def test_store_procedural_with_trigger_context(db_session, embedding_service):
    mem = store_procedural(
        db_session, embedding_service,
        content="comfort procedure",
        trigger_context="someone is sad",
    )
    assert mem.metadata_["trigger_context"] == "someone is sad"


def test_store_procedural_with_steps(db_session, embedding_service):
    mem = store_procedural(
        db_session, embedding_service,
        content="problem solving",
        steps=["identify", "analyze", "solve"],
    )
    assert mem.metadata_["steps"] == ["identify", "analyze", "solve"]
