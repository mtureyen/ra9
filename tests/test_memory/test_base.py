"""Tests for base memory operations (store and recall)."""

import uuid

from emotive.db.models.memory import Memory
from emotive.memory.base import recall_memories, store_memory
from emotive.runtime.event_bus import MEMORY_RECALLED, MEMORY_STORED


def test_store_memory_creates_row(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="test memory content", memory_type="episodic",
    )
    db_session.flush()
    assert mem.id is not None
    assert mem.memory_type == "episodic"
    assert mem.content == "test memory content"


def test_store_memory_generates_embedding(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="embedding test", memory_type="episodic",
    )
    assert mem.embedding is not None
    assert len(list(mem.embedding)) == 1024


def test_store_memory_sets_decay_rate(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="decay test", memory_type="episodic", decay_rate=0.005,
    )
    assert mem.decay_rate == 0.005


def test_store_memory_publishes_event(db_session, embedding_service, event_bus):
    received = []
    event_bus.subscribe(MEMORY_STORED, lambda t, d: received.append(t))
    store_memory(
        db_session, embedding_service,
        content="event test", memory_type="episodic", event_bus=event_bus,
    )
    assert len(received) == 1


def test_store_memory_without_event_bus(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="no bus", memory_type="episodic", event_bus=None,
    )
    assert mem.id is not None


def test_recall_memories_returns_ranked_results(db_session, embedding_service):
    store_memory(db_session, embedding_service, content="dogs in the park", memory_type="episodic")
    store_memory(db_session, embedding_service, content="cats on the roof", memory_type="episodic")
    store_memory(db_session, embedding_service, content="quantum physics", memory_type="episodic")
    db_session.flush()

    results = recall_memories(
        db_session, embedding_service, query="dogs playing outside", limit=3,
    )
    assert len(results) > 0
    scores = [r["final_rank"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_recall_memories_updates_retrieval_count(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="retrieval count test unique xyz", memory_type="episodic",
    )
    db_session.flush()
    assert mem.retrieval_count == 0

    recall_memories(
        db_session, embedding_service,
        query="retrieval count test unique xyz", limit=1,
    )
    db_session.flush()
    db_session.refresh(mem)
    assert mem.retrieval_count >= 1


def test_recall_memories_marks_labile(db_session, embedding_service):
    mem = store_memory(
        db_session, embedding_service,
        content="labile test unique content abc", memory_type="episodic",
    )
    db_session.flush()

    recall_memories(
        db_session, embedding_service,
        query="labile test unique content abc", limit=1,
    )
    db_session.flush()
    db_session.refresh(mem)
    assert mem.is_labile is True
    assert mem.labile_until is not None


def test_recall_memories_empty_db(db_session, embedding_service):
    results = recall_memories(
        db_session, embedding_service, query="nothing here", limit=5,
    )
    # May return results from other tests' data, but should not error
    assert isinstance(results, list)


def test_recall_memories_publishes_events(db_session, embedding_service, event_bus):
    store_memory(
        db_session, embedding_service,
        content="event recall test unique qqq", memory_type="episodic",
    )
    db_session.flush()

    received = []
    event_bus.subscribe(MEMORY_RECALLED, lambda t, d: received.append(t))
    recall_memories(
        db_session, embedding_service,
        query="event recall test unique qqq", limit=3, event_bus=event_bus,
    )
    assert len(received) >= 1
